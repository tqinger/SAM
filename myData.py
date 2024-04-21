import random

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
join = os.path.join
from tqdm import tqdm

import os
from PIL import Image,ImageDraw


import transforms as T

import torch

from torch.utils.data import Dataset, DataLoader

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(1 * base_size)
        max_size = int(1.0 * base_size)
        trans = []
        trans = [T.RandomResize(min_size, max_size)]

        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            T.ToTensor(),
            T.CenterCrop(crop_size),

            # T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.CenterCrop(base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):

        return self.transforms(img, target)


def get_transform(train):
    # base_size = 520
    # crop_size = 480
    base_size = 1024
    crop_size = 1024

    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(base_size)


class VOCSegmentation(Dataset):
    def __init__(self, voc_root, bbox_txt_file, year="2012",bbox_shift=20,transforms=None, txt_name: str = "train.txt"):
        super(VOCSegmentation, self).__init__()
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClass')
        self.bbox_txt_file = bbox_txt_file
        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        self.file_names = file_names
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.bbox_txt_path = [os.path.join(self.bbox_txt_file,x + ".txt") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms
        self.bbox_shift = bbox_shift

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        useless = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        wide,height = target.size
        print(wide,height)
        img_name = self.file_names[index]
        bbox_txt = self.bbox_txt_path[index]

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        with open(bbox_txt, "r") as f:
            first_line = f.readline()
            x_min, y_min, x_max, y_max = first_line.split(" ")

            x_min = (int(x_min)/wide)*1024
            x_max = (int(x_max)/wide) * 1024
            y_min = (int(y_min)/height)*1024
            y_max = (int(y_max)/height)*1024
        # 输出结果为Tensor格式
        bboxes = torch.tensor([int(x_min), int(y_min), int(x_max), int(y_max)], dtype=torch.float32)
        return (
        img,
        torch.tensor(target[None, :, :]).long(),
        bboxes,
        img_name,
    )
    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    npy_path = r"C:\Users\Public\data\shetou"
    train_data = VOCSegmentation(voc_root=npy_path,bbox_txt_file=r"C:\Users\Public\data\shetou\VOCdevkit\VOC2012\Bbox",
                                 transforms=get_transform(train=True))
    a,b,c,d = train_data[20]
    print(a.shape)
    # print(b.shape)
    print(c,c.shape)
    # 将Tensor数据转换为PIL图像
    # 首先，确保Tensor的数据类型是float32，并且范围在[0, 1]（如果不是，请转换）
    # 然后，将其转换为[0, 255]的uint8范围，再转换为numpy数组，并最后转为PIL图像
    img_pil = Image.fromarray((a.permute(1, 2, 0).numpy()*255).astype(np.uint8))

    # 使用PIL在图像上画线
    draw = ImageDraw.Draw(img_pil)
    # 画一条从(50, 50)到(200, 200)的线，线条颜色为红色，宽度为5
    draw.line((c[0], c[1], c[0], c[3]), fill='red', width=5)
    draw.line((c[0], c[3], c[2], c[3]), fill='red', width=5)
    draw.line((c[2], c[3], c[2], c[1]), fill='red', width=5)
    draw.line((c[0], c[1], c[2], c[1]), fill='red', width=5)

    # 显示图像
    img_pil.show()
    #
    # print(d)

