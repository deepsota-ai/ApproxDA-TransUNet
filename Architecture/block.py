import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import numpy as np
from os.path import join as pjoin
from collections import OrderedDict


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(
        cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups
    )


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block."""

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(
            cmid, cmid, stride, bias=False
        )  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, "downsample"):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, "downsample"):
            proj_conv_weight = np2th(
                weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True
            )
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        StdConv2d(
                            3, width, kernel_size=7, stride=2, bias=False, padding=3
                        ),
                    ),
                    ("gn", nn.GroupNorm(32, width, eps=1e-6)),
                    ("relu", nn.ReLU(inplace=True)),
                    # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
                ]
            )
        )

        self.body = nn.Sequential(
            OrderedDict(
                [
                    (
                        "block1",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width, cout=width * 4, cmid=width
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 4, cout=width * 4, cmid=width
                                        ),
                                    )
                                    for i in range(2, block_units[0] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block2",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width * 4,
                                            cout=width * 8,
                                            cmid=width * 2,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 8,
                                            cout=width * 8,
                                            cmid=width * 2,
                                        ),
                                    )
                                    for i in range(2, block_units[1] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block3",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width * 8,
                                            cout=width * 16,
                                            cmid=width * 4,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 16,
                                            cout=width * 16,
                                            cmid=width * 4,
                                        ),
                                    )
                                    for i in range(2, block_units[2] + 1)
                                ],
                            )
                        ),
                    ),
                ]
            )
        )

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(
                    x.size(), right_size
                )
                feat = torch.zeros(
                    (b, x.size()[1], right_size, right_size), device=x.device
                )
                feat[:, :, 0 : x.size()[2], 0 : x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


# ---------------------------------------------------------------------------
# ApproxDA-TransUNet: windowed low-rank PAM, grouped CAM, adaptive DA block
# ---------------------------------------------------------------------------


def window_partition(x, window_size):
    """(B, C, H, W) -> (B*nW, C, M, M)"""
    B, C, H, W = x.shape
    M = window_size
    x = x.contiguous().view(B, C, H // M, M, W // M, M)
    return x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, M, M)


def window_reverse(windows, window_size, H, W):
    """(B*nW, C, M, M) -> (B, C, H, W)"""
    M = window_size
    nW = (H // M) * (W // M)
    B = windows.shape[0] // nW
    C = windows.shape[1]
    x = windows.contiguous().view(B, H // M, W // M, C, M, M)
    return x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)


def exact_window_attention(q, k, v):
    """softmax(q^T k) v^T without scaling, as PAM_Module: q, k (B, d, N), v (B, C, N) -> (B, N, C).

    On CUDA this runs the memory-efficient SDPA kernel, which never stores the N x N
    scores (15 GB at M=112, bs 24). The kernel needs head dims divisible by 4, so q/k/v
    are zero-padded (q^T k is unchanged; padded value channels are sliced off), and it is
    forced so an unsupported case raises instead of falling back to the N x N math path.
    On CPU (fvcore FLOP audits) the explicit matmuls are kept.
    """
    if not q.is_cuda:
        scores = F.softmax(torch.bmm(q.transpose(1, 2), k), dim=-1)
        return torch.bmm(scores, v.transpose(1, 2))
    C = v.shape[1]
    pad_qk, pad_v = -q.shape[1] % 4, -C % 4
    q = F.pad(q.transpose(1, 2), (0, pad_qk)).contiguous().unsqueeze(1)  # (B, 1, N, d')
    k = F.pad(k.transpose(1, 2), (0, pad_qk)).contiguous().unsqueeze(1)
    v = F.pad(v.transpose(1, 2), (0, pad_v)).contiguous().unsqueeze(1)  # (B, 1, N, C')
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        out = F.scaled_dot_product_attention(q, k, v, scale=1.0)
    return out.squeeze(1)[..., :C]


class LowRankWindowedPAM(nn.Module):
    """
    Windowed PAM: O(N^2 C) -> O(N*r*C) via Swin-style windows + low-rank projection.
    Keys and values are projected from window_size^2 -> rank via self.proj_r.
    """

    def __init__(self, channels, window_size=7, rank=32, qk_channels=None):
        super().__init__()
        self.M = window_size
        N = window_size**2
        # ApproxDABlock passes channels // 8 query/key channels, as DA-TransUNet's PAM_Module.
        qk = channels if qk_channels is None else qk_channels
        self.conv_B = nn.Conv1d(channels, qk, 1)
        self.conv_C = nn.Conv1d(channels, qk, 1)
        self.conv_D = nn.Conv1d(channels, channels, 1)
        # rank <= 0: no low-rank projection, i.e. exact attention inside each window
        self.proj_r = nn.Linear(N, rank, bias=False) if rank > 0 else None
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        # Clamp window to feature-map size so window_size=112 gives global attention
        # at every scale without requiring H to be a multiple of self.M.
        M = min(self.M, H, W)
        x_w = window_partition(x, M)  # (B*nW, C, M, M)
        nBW = x_w.shape[0]
        x_n = x_w.view(nBW, C, M * M)  # (B*nW, C, N_actual)
        feat_B = self.conv_B(x_n)
        feat_C = self.conv_C(x_n)
        feat_D = self.conv_D(x_n)
        # proj_r was registered with N=self.M**2; slice weights when clamped
        N_actual = M * M
        if self.proj_r is None:
            # exact window attention
            E_out = exact_window_attention(feat_B, feat_C, feat_D)  # (B*nW, N, C)
        else:
            if N_actual == self.M**2:
                C_r = self.proj_r(feat_C)  # (B*nW, C, r)
                D_r = self.proj_r(feat_D)
            else:
                w = self.proj_r.weight[:, :N_actual]  # (r, N_actual)
                C_r = F.linear(feat_C, w)
                D_r = F.linear(feat_D, w)
            scores = torch.bmm(feat_B.transpose(1, 2), C_r)  # (B*nW, N, r)
            scores = F.softmax(scores, dim=-1)
            E_out = torch.bmm(scores, D_r.transpose(1, 2))  # (B*nW, N, C)
        E_n = self.alpha * E_out.transpose(1, 2) + x_n  # (B*nW, C, N)
        return window_reverse(E_n.view(nBW, C, M, M), M, H, W)


class GroupedCAM(nn.Module):
    """
    Channel attention split into G independent groups: O(C^2) -> O(C^2/G).
    Each group computes its own (C/G x C/G) attention matrix.
    """

    def __init__(self, channels, groups=8):
        super().__init__()
        # Narrow maps (e.g. 64/16 = 4 channels in the C/16 block) cannot hold
        # more groups than channels; fall back to one channel per group.
        self.G = min(groups, channels)
        assert channels % self.G == 0, f"{channels} channels not divisible by {self.G} groups"
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        G, Cg = self.G, C // self.G
        x_g = x.contiguous().view(B * G, Cg, H * W)  # (B*G, Cg, N)
        X = torch.bmm(x_g, x_g.transpose(1, 2))  # (B*G, Cg, Cg)
        # Match DA-TransUNet CAM_Module: softmax(max - energy), then aggregate
        # over the normalized axis, so G=1 reproduces the original CAM exactly.
        X = torch.max(X, -1, keepdim=True)[0].expand_as(X) - X
        X = F.softmax(X, dim=-1)
        E_g = torch.bmm(X, x_g)  # (B*G, Cg, N)
        E = self.beta * E_g + x_g
        return E.contiguous().view(B, C, H, W)


def norm(planes):
    """BatchNorm with the same settings as DA-TransUNet's DANetHead."""
    return nn.BatchNorm2d(planes, momentum=0.95, eps=1e-03)


def conv_bn_relu(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), norm(cout), nn.ReLU()
    )


class ApproxDABlock(nn.Module):
    """
    ApproxDA Block with DA-TransUNet's C/16 bottleneck (DANetHead layout).

    Identical to DANetHead except that PAM/CAM are replaced by
    LowRankWindowedPAM / GroupedCAM and the two branches are fused by a gate:
      x -> conv5a (3x3, C->C/16) -> windowed low-rank PAM -> conv51 (3x3) --\
      x -> conv5c (3x3, C->C/16) -> grouped CAM           -> conv52 (3x3) --+-> g-fusion
        -> conv8 (Dropout2d 0.05, 1x1 C/16->C, ReLU)          (no outer residual, as DA)
    PAM queries/keys use (C/16)/8 channels as in DA's PAM_Module (min 1).
    DANetHead's conv6/conv7 are omitted: their outputs are discarded in DA.
    gate_mode: 'pam' (g=1) / 'cam' (g=0) build only the branch they use;
    'fixed' g=0.5; 'learn' g = sigmoid(Linear(GAP(x))) per C/16 channel.
    """

    def __init__(self, channels, window_size=7, rank=32, groups=8, gate_mode="learn"):
        super().__init__()
        inter = channels // 16
        self.gate_mode = gate_mode
        self.use_pam = gate_mode != "cam"
        self.use_cam = gate_mode != "pam"
        if self.use_pam:
            self.conv5a = conv_bn_relu(channels, inter)
            self.pam = LowRankWindowedPAM(
                inter, window_size, rank, qk_channels=max(1, inter // 8)
            )
            self.conv51 = conv_bn_relu(inter, inter)
        if self.use_cam:
            self.conv5c = conv_bn_relu(channels, inter)
            self.cam = GroupedCAM(inter, groups)
            self.conv52 = conv_bn_relu(inter, inter)
        if gate_mode == "learn":
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.gate_fc = nn.Linear(channels, inter)
        self.conv8 = nn.Sequential(
            nn.Dropout2d(0.05, False), nn.Conv2d(inter, channels, 1), nn.ReLU()
        )

    def forward(self, x):
        if self.gate_mode == "pam":
            fused = self.conv51(self.pam(self.conv5a(x)))
        elif self.gate_mode == "cam":
            fused = self.conv52(self.cam(self.conv5c(x)))
        else:
            p = self.conv51(self.pam(self.conv5a(x)))
            c = self.conv52(self.cam(self.conv5c(x)))
            if self.gate_mode == "fixed":
                g = 0.5
            else:  # 'learn'
                B = x.shape[0]
                g = torch.sigmoid(self.gate_fc(self.pool(x).view(B, -1))).view(B, -1, 1, 1)
            fused = g * p + (1.0 - g) * c
        return self.conv8(fused)
