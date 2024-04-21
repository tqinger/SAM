import torch.utils.data as data
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
from torch.utils.data import Dataset, DataLoader
import random
import transforms as T

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
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):

        return self.transforms(img, target)


def get_transform(train):
    # base_size = 520
    # crop_size = 480
    base_size = 1200
    crop_size = 1200

    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(base_size)


class VOCSegmentation(Dataset):
    def __init__(self, voc_root, year="2012", bbox_shift=20,transforms=None, txt_name: str = "train.txt"):
        super(VOCSegmentation, self).__init__()
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClass')

        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        self.file_names = file_names
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
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
        target = Image.open(self.masks[index])
        img_name = self.file_names[index]

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # gt2D = np.array(target)
        # y_indices, x_indices = np.where(gt2D > 0)
        # x_min, x_max = np.min(x_indices), np.max(x_indices)
        # y_min, y_max = np.min(y_indices), np.max(y_indices)
        # # add perturbation to bounding box coordinates
        # H, W = gt2D.shape
        # x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        # x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        # y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        # y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        # bboxes = np.array([x_min, y_min, x_max, y_max])
        gt2D = target

        # 使用PyTorch找到gt2D中大于0的位置的索引
        y_indices, x_indices = (gt2D > 0).nonzero(as_tuple=True)

        # 使用PyTorch的操作找到最大值和最小值
        x_min, x_max = torch.min(x_indices), torch.max(x_indices)
        y_min, y_max = torch.min(y_indices), torch.max(y_indices)

        # 添加扰动到边界框坐标
        H, W = gt2D.shape
        x_min = torch.clamp(x_min - random.randint(0, self.bbox_shift), min=0)
        x_max = torch.clamp(x_max + random.randint(0, self.bbox_shift), max=W - 1)
        y_min = torch.clamp(y_min - random.randint(0, self.bbox_shift), min=0)
        y_max = torch.clamp(y_max + random.randint(0, self.bbox_shift), max=H - 1)

        # 输出结果为Tensor格式
        bboxes = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
        return (
        img,
        torch.tensor(gt2D[None, :, :]).long(),
        bboxes,
        img_name,
    )



    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


# dataset = VOCSegmentation(voc_root="/data/", transforms=get_transform(train=True))
# d1 = dataset[0]
# print(d1)
if __name__ == "__main__":
    train_dataset = VOCSegmentation(voc_root=r"C:\Users\Public\data\she\514_yuantu",
                                    year="2012",
                                    transforms=get_transform(train=True),
                                    txt_name="train.txt")
    # tr_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    a,b,c,d = train_dataset[0]
    print(a.shape)
    print(b.shape)
    print(c,c.shape)

    print(d)
    # for step, (image, gt, bboxes, names_temp) in enumerate(tr_dataloader):
    #     print(image.shape, gt.shape, bboxes.shape)
    #     # show the example
    #     _, axs = plt.subplots(1, 2, figsize=(25, 25))
    #     idx = random.randint(0, 7)
    #     axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    #     show_mask(gt[idx].cpu().numpy(), axs[0])
    #     show_box(bboxes[idx].numpy(), axs[0])
    #     axs[0].axis("off")
    #     # set title
    #     axs[0].set_title(names_temp[idx])
    #     idx = random.randint(0, 7)
    #     axs[1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    #     show_mask(gt[idx].cpu().numpy(), axs[1])
    #     show_box(bboxes[idx].numpy(), axs[1])
    #     axs[1].axis("off")
    #     # set title
    #     axs[1].set_title(names_temp[idx])
    #     # plt.show()
    #     plt.subplots_adjust(wspace=0.01, hspace=0)
    #     plt.savefig("./data_sanitycheck.png", bbox_inches="tight", dpi=300)
    #     plt.close()