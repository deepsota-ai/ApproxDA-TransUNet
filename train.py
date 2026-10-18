"""Train ApproxDA-TransUNet on Synapse, CVC-ClinicDB or ISIC 2018 (single GPU).

The final-epoch checkpoint is the one evaluated by test.py; no checkpoint is selected on test data.
"""

import argparse
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from Architecture.ApproxDATransUNet import ApproxDATransUNet
from Architecture.ApproxDATransUNet import CONFIGS as CONFIGS_ViT_seg
from utils import DiceLoss

DATASETS = {
    "Synapse": dict(root_path="./data/Synapse/train_npz", list_dir="./lists/lists_Synapse", num_classes=9),
    "CVC": dict(root_path="./data/CVC-ClinicDB", list_dir="./lists/lists_CVC", num_classes=2),
    "ISIC": dict(root_path="./data/ISIC2018", list_dir="./lists/lists_ISIC", num_classes=2),
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="Synapse", choices=list(DATASETS))
parser.add_argument("--root_path", type=str, default=None, help="training data dir (default: see DATASETS)")
parser.add_argument("--max_epochs", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--base_lr", type=float, default=0.01)
parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
parser.add_argument("--img_size", type=int, default=224, help="network input size")
parser.add_argument("--rgb", action="store_true", help="RGB input (CVC/ISIC); grayscale otherwise")
parser.add_argument("--drop_last", action="store_true", help="drop the incomplete last batch of each epoch")
parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers (changes the augmentation stream)")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--deterministic", type=int, default=1, help="cuDNN deterministic mode")
parser.add_argument("--n_skip", type=int, default=3, help="number of skip connections")
parser.add_argument("--vit_name", type=str, default="R50-ViT-B_16")
parser.add_argument("--vit_patches_size", type=int, default=16)
parser.add_argument("--window_size", type=int, default=7, help="PAM window size M")
parser.add_argument("--rank", type=int, default=32, help="PAM projection rank r (0 = exact attention)")
parser.add_argument("--groups", type=int, default=8, help="CAM channel groups G")
parser.add_argument("--gate_mode", type=str, default="pam", choices=["pam", "cam", "fixed", "learn"])
parser.add_argument("--output_dir", type=str, default="./model")
args = parser.parse_args()


def build_optimizer(args, model):
    if args.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), weight_decay=0.0001)
    return optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)


def build_train_set(args):
    size = [args.img_size, args.img_size]
    if args.dataset == "Synapse":
        from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

        return Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose([RandomGenerator(output_size=size)]))
    if args.dataset == "CVC":
        from datasets.dataset_cvc import CVC_dataset as Dataset, RandomGenerator
    else:
        from datasets.dataset_isic import ISIC_dataset as Dataset, RandomGenerator
    return Dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                   transform=transforms.Compose([RandomGenerator(size)]), **({"rgb": True} if args.rgb else {}))


def train(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format="[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    db_train = build_train_set(args)
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                             drop_last=args.drop_last, pin_memory=True, worker_init_fn=worker_init_fn)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)
    optimizer = build_optimizer(args, model)
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    train_start = time.time()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        epoch_loss_sum, epoch_loss_ce_sum, epoch_batches = 0.0, 0.0, 0
        for sampled_batch in trainloader:
            image_batch, label_batch = sampled_batch["image"].cuda(), sampled_batch["label"].cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # poly learning-rate decay
            lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_
            iter_num = iter_num + 1
            epoch_loss_sum += loss.item()
            epoch_loss_ce_sum += loss_ce.item()
            epoch_batches += 1
        logging.info("epoch %d/%d  loss: %.4f  loss_ce: %.4f  lr: %.6f  elapsed: %.2fh"
                     % (epoch_num + 1, max_epoch, epoch_loss_sum / epoch_batches,
                        epoch_loss_ce_sum / epoch_batches, lr_, (time.time() - train_start) / 3600))
    save_path = os.path.join(snapshot_path, "epoch_{}.pth".format(max_epoch - 1))
    torch.save(model.state_dict(), save_path)
    logging.info("save model to {}".format(save_path))
    return "Training Finished!"


if __name__ == "__main__":
    # The DataLoader worker_init_fn is a closure; Python >= 3.14 defaults to forkserver, which cannot pickle it.
    torch.multiprocessing.set_start_method("fork")
    cudnn.benchmark = not args.deterministic
    cudnn.deterministic = bool(args.deterministic)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    cfg = DATASETS[args.dataset]
    args.num_classes = cfg["num_classes"]
    args.root_path = args.root_path or cfg["root_path"]
    args.list_dir = cfg["list_dir"]
    name = f"{args.dataset}{args.img_size}_M{args.window_size}_r{args.rank}_G{args.groups}_{args.gate_mode}_s{args.seed}"
    snapshot_path = os.path.join(args.output_dir, name)
    os.makedirs(snapshot_path, exist_ok=True)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.window_size = args.window_size
    config_vit.rank = args.rank
    config_vit.groups = args.groups
    config_vit.gate_mode = args.gate_mode
    if args.vit_name.find("R50") != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ApproxDATransUNet(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    net.load_from(weights=np.load(config_vit.pretrained_path))
    train(args, net, snapshot_path)
