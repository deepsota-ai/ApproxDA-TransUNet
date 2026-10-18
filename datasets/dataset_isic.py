import os
import random
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import zoom
import torch
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=3, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape[:2]
        if x != self.output_size[0] or y != self.output_size[1]:
            scale = (self.output_size[0] / x, self.output_size[1] / y)
            image = zoom(image, scale + (1,) * (image.ndim - 2), order=3)
            label = zoom(
                label, (self.output_size[0] / x, self.output_size[1] / y), order=0
            )
        image = torch.from_numpy(image.astype(np.float32))
        # (H, W) grayscale -> (1, H, W); (H, W, 3) RGB -> (3, H, W)
        image = image.permute(2, 0, 1) if image.ndim == 3 else image.unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        return {"image": image, "label": label.long()}


class ISIC_dataset(Dataset):
    """
    ISIC 2018 Task 1 skin-lesion segmentation dataset.

    Directory layout expected under base_dir:
        images/   — JPEG images named ISIC_XXXXXXX.jpg
        masks/    — PNG masks  named ISIC_XXXXXXX_segmentation.png

    List files (one stem per line, WITHOUT extension and WITHOUT _segmentation suffix):
        {list_dir}/train.txt   — e.g.  ISIC_0024306
        {list_dir}/test.txt

    test.py calls with split='test_vol'; we remap it to 'test' transparently.
    """

    def __init__(self, base_dir, list_dir, split, transform=None, rgb=False):
        self.transform = transform
        self.rgb = rgb  # True: (H, W, 3) RGB input; False: grayscale (default)
        list_split = "test" if split == "test_vol" else split
        self.sample_list = [
            l
            for l in open(os.path.join(list_dir, list_split + ".txt")).readlines()
            if l.strip() and not l.strip().startswith("#")
        ]
        self.img_dir = os.path.join(base_dir, "images")
        self.mask_dir = os.path.join(base_dir, "masks")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx].strip("\n")
        image = (
            np.array(
                Image.open(os.path.join(self.img_dir, name + ".jpg")).convert("RGB" if self.rgb else "L"),
                dtype=np.float32,
            )
            / 255.0
        )
        label = (
            np.array(
                Image.open(
                    os.path.join(self.mask_dir, name + "_segmentation.png")
                ).convert("L")
            )
            > 127
        ).astype(np.uint8)

        sample = {"image": image, "label": label}
        if self.transform:
            sample = self.transform(sample)
        sample["case_name"] = name
        return sample
