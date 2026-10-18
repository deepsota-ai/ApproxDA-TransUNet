"""Evaluate an ApproxDA-TransUNet checkpoint on the test split of Synapse, CVC-ClinicDB or ISIC 2018.

Reports per-image (per-volume for Synapse) mean Dice, HD95 and IoU, averaged over foreground classes.
"""

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from Architecture.ApproxDATransUNet import ApproxDATransUNet
from Architecture.ApproxDATransUNet import CONFIGS as CONFIGS_ViT_seg
from datasets.dataset_cvc import CVC_dataset
from datasets.dataset_isic import ISIC_dataset
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume

DATASETS = {
    "Synapse": dict(Dataset=Synapse_dataset, volume_path="./data/Synapse/test_vol_h5",
                    list_dir="./lists/lists_Synapse", num_classes=9),
    "CVC": dict(Dataset=CVC_dataset, volume_path="./data/CVC-ClinicDB", list_dir="./lists/lists_CVC", num_classes=2),
    "ISIC": dict(Dataset=ISIC_dataset, volume_path="./data/ISIC2018", list_dir="./lists/lists_ISIC", num_classes=2),
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="Synapse", choices=list(DATASETS))
parser.add_argument("--checkpoint", type=str, required=True, help="path to the .pth state dict")
parser.add_argument("--volume_path", type=str, default=None, help="test data dir (default: see DATASETS)")
parser.add_argument("--img_size", type=int, default=224, help="network input size")
parser.add_argument("--rgb", action="store_true", help="RGB input (CVC/ISIC); grayscale otherwise")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--deterministic", type=int, default=1)
parser.add_argument("--n_skip", type=int, default=3)
parser.add_argument("--vit_name", type=str, default="R50-ViT-B_16")
parser.add_argument("--vit_patches_size", type=int, default=16)
parser.add_argument("--window_size", type=int, default=7, help="PAM window size M")
parser.add_argument("--rank", type=int, default=32, help="PAM projection rank r (0 = exact attention)")
parser.add_argument("--groups", type=int, default=8, help="CAM channel groups G")
parser.add_argument("--gate_mode", type=str, default="pam", choices=["pam", "cam", "fixed", "learn"])
parser.add_argument("--log_dir", type=str, default="./test_log")
args = parser.parse_args()


def inference(args, model):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir,
                           **({"rgb": True} if args.rgb else {}))
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test cases".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch["case_name"][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes,
                                      patch_size=[args.img_size, args.img_size])
        metric_list += np.array(metric_i)
        logging.info("idx %d case %s mean_dice %f mean_hd95 %f"
                     % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info("Mean class %d mean_dice %f mean_hd95 %f mean_iou %f"
                     % (i, metric_list[i - 1][0], metric_list[i - 1][1], metric_list[i - 1][2]))
    mean = np.mean(metric_list, axis=0)
    logging.info("Testing performance: mean_dice : %f mean_hd95 : %f mean_iou : %f" % (mean[0], mean[1], mean[2]))
    return "Testing Finished!"


if __name__ == "__main__":
    cudnn.benchmark = not args.deterministic
    cudnn.deterministic = bool(args.deterministic)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    cfg = DATASETS[args.dataset]
    args.Dataset = cfg["Dataset"]
    args.num_classes = cfg["num_classes"]
    args.volume_path = args.volume_path or cfg["volume_path"]
    args.list_dir = cfg["list_dir"]

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.window_size = args.window_size
    config_vit.rank = args.rank
    config_vit.groups = args.groups
    config_vit.gate_mode = args.gate_mode
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find("R50") != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ApproxDATransUNet(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    net.load_state_dict({k.removeprefix("module."): v for k, v in state_dict.items()}, strict=True)

    os.makedirs(args.log_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    logging.basicConfig(filename=os.path.join(args.log_dir, f"{args.dataset}_{name}.txt"), level=logging.INFO,
                        format="[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    inference(args, net)
