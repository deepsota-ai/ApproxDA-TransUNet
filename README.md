# ApproxDA-TransUNet

Code and trained weights for

> **ApproxDA-TransUNet: Understanding Global Context Sensitivity of Attention Approximation for Medical Image Segmentation**
> Jiawei Zhong, Qin Chen, Hui Zhang. IEEE BIBM 2026.

ApproxDA-TransUNet keeps the hybrid R50-ViT-B/16 encoder–decoder of DA-TransUNet and replaces its dual-attention
heads with ApproxDABlocks: a windowed low-rank position attention (window size `M`, projection rank `r`) and a
grouped channel attention (`G` groups), combined by a configurable gate (`pam`, `cam`, `fixed`, `learn`).
ApproxDABlocks sit at the bottleneck and at all three skip connections.

## Results

Single run (seed 1234), final-epoch checkpoint, test split.

| Dataset | Input | Configuration | DSC (%) | |
|---|---|---|---|---|
| Synapse | 224, grayscale | gate=pam, M=112, r=32, G=8 | 81.63 | HD95 23.73 mm |
| CVC-ClinicDB | 256, grayscale | gate=cam, G=1 | 90.54 | mIoU 85.87 % |
| ISIC 2018 | 256, RGB | gate=pam, M=64, r=32, G=8 | 89.26 | mIoU 82.64 % |

## Setup

Tested with Python 3.14, PyTorch 2.14 (CUDA 13.0), torchvision 0.29, NumPy 2.5, SciPy 1.18 on one NVIDIA RTX 5090.

```bash
pip install torch torchvision numpy scipy h5py medpy ml_collections pillow tqdm
```

ImageNet-21k R50-ViT-B/16 weights (needed for training only):

```bash
mkdir -p pretrained/imagenet21k
wget -P pretrained/imagenet21k https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz
```

## Data

Place the data under `./data` (or pass `--root_path` / `--volume_path`). The train/test splits used in the paper
are in `lists/`.

| Dataset | Layout | Split |
|---|---|---|
| Synapse | `data/Synapse/train_npz/*.npz`, `data/Synapse/test_vol_h5/*.npy.h5` (preprocessed data of [TransUNet](https://github.com/Beckschen/TransUNet)) | 18 / 12 CT volumes |
| CVC-ClinicDB | `data/CVC-ClinicDB/PNG/Original/<id>.png`, `data/CVC-ClinicDB/PNG/Ground Truth/<id>.png` | random 75/25: 459 / 153 frames |
| ISIC 2018 | `data/ISIC2018/images/<id>.jpg`, `data/ISIC2018/masks/<id>_segmentation.png` | random 75/25 of the 2594 Task 1 training images: 1945 / 649 |

ISIC 2018 images and masks are resized once to 256×256 before training and evaluation (metrics are computed at
this resolution):

```python
from PIL import Image
img.convert("RGB").resize((256, 256), Image.BICUBIC).save(dst_jpg, quality=100, subsampling=0)  # images
mask.convert("L").resize((256, 256), Image.NEAREST).save(dst_png)                                # masks
```

## Trained weights

Download from the [releases page](https://github.com/deepsota-ai/ApproxDA-TransUNet/releases) into `weights/`.

| File | Dataset | SHA-256 |
|---|---|---|
| `approxda_synapse.pth` | Synapse | `7b8f2fecb4b52268379e38a7ac5affb121f87dfdbbfc3134700a46435536f363` |
| `approxda_cvc.pth` | CVC-ClinicDB | `f263aec2acea1ff9111e9dd9ebbd407501bbc80b30fa58c5acdf28f5ffde2e98` |
| `approxda_isic.pth` | ISIC 2018 | `640d01be20a85ddb81818b8d29e2a1d62c8f82fd10b438eb875a6db034a4a155` |

## Evaluation

```bash
python test.py --dataset Synapse --checkpoint weights/approxda_synapse.pth --img_size 224 \
    --window_size 112 --rank 32 --groups 8 --gate_mode pam
python test.py --dataset CVC --checkpoint weights/approxda_cvc.pth --img_size 256 \
    --window_size 32 --rank 32 --groups 1 --gate_mode cam
python test.py --dataset ISIC --checkpoint weights/approxda_isic.pth --img_size 256 --rgb \
    --window_size 64 --rank 32 --groups 8 --gate_mode pam
```

Per-case and per-class results are written to `test_log/`.

## Training

```bash
# Synapse: SGD (lr 0.01, poly decay), 300 epochs
python train.py --dataset Synapse --max_epochs 300 --img_size 224 \
    --window_size 112 --rank 32 --groups 8 --gate_mode pam
# CVC-ClinicDB: Adam (lr 1e-3), 500 epochs
python train.py --dataset CVC --max_epochs 500 --img_size 256 --optimizer adam --base_lr 0.001 --drop_last \
    --window_size 32 --rank 32 --groups 1 --gate_mode cam
# ISIC 2018: SGD (lr 0.01), 50 epochs
python train.py --dataset ISIC --max_epochs 50 --img_size 256 --rgb --drop_last \
    --window_size 64 --rank 32 --groups 8 --gate_mode pam
```

All runs use batch size 24 and seed 1234; the final-epoch checkpoint is saved to `model/<run>/` and evaluated with
`test.py --checkpoint model/<run>/epoch_<max_epochs-1>.pth`. GPU training is not bitwise deterministic, so a re-run
differs slightly from the released weights. `--rank 0` gives exact attention inside each window.

## Acknowledgements

This code builds on [TransUNet](https://github.com/Beckschen/TransUNet) and
[DA-TransUNet](https://github.com/sun-1024/da-transunet).

## Citation

```bibtex
@inproceedings{zhong2026approxda,
  title     = {ApproxDA-TransUNet: Understanding Global Context Sensitivity of Attention Approximation for Medical Image Segmentation},
  author    = {Zhong, Jiawei and Chen, Qin and Zhang, Hui},
  booktitle = {IEEE International Conference on Bioinformatics and Biomedicine (BIBM) Workshops},
  year      = {2026}
}
```
