# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
import transforms as T
# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


def iou_score(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / union if union != 0 else 0

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
    base_size = 1024
    crop_size = 1024

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



class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        label_ids = np.unique(gt)[1:]
        gt2D = np.uint8(
            gt == random.choice(label_ids.tolist())
        )  # only one label, (256, 256)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )


# # %% sanity test of dataset class
# tr_dataset = NpyDataset("data/npy/CT_Abd")
# tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)
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
#     break

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    default="data/npy/CT_Abd",
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-checkpoint", type=str, default=r"C:\Users\Public\cv\sam\playground\label_anything\sam_vit_b_01ec64.pth"
)
# parser.add_argument('-device', type=str, default='cuda:0')
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="use wandb to monitor training"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=1000)
parser.add_argument("-batch_size", type=int, default=2)
parser.add_argument("-num_workers", type=int, default=0)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument(
    "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

if args.use_wandb:
    import wandb

    wandb.login()
    wandb.init(
        project=args.task_name,name="sam_b",
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_npy_path,
            "model_type": args.model_type,
        },
    )

# %% set up model for training
# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)
# %% set up model


class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in medsam_model.parameters()),
    )  # 93735472
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
    )  # 93729252

    img_mask_encdec_params = list(medsam_model.image_encoder.parameters()) + list(
        medsam_model.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )  # 93729252
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # %% train
    num_epochs = args.num_epochs
    iter_num = 0
    losses = []
    best_loss = 1e10
    train_dataset = VOCSegmentation(voc_root=args.tr_npy_path,
                                    year="2012",
                                    transforms=get_transform(train=True),
                                    txt_name="train.txt")

    val_dataset = VOCSegmentation(voc_root=args.tr_npy_path,
                                    year="2012",
                                    transforms=get_transform(train=False),
                                    txt_name="val.txt")

    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        collate_fn=val_dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # 假设以下函数和变量已经定义: seg_loss, ce_loss, train_dataloader, val_dataloader, medsam_model, optimizer, device, args, model_save_path

    # iou_score 的定义如上所述


    best_loss = float('inf')
    best_iou = 0.0
    iter_num = 0
    losses = []

    # 开始训练循环
    for epoch in range(start_epoch, num_epochs):
        medsam_model.train()
        epoch_loss = 0
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)

            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = medsam_model(image, boxes_np)
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                medsam_pred = medsam_model(image, boxes_np)
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            iter_num += 1

        epoch_loss /= len(train_dataloader)  # 修正：使用数据集的长度进行除法
        losses.append(epoch_loss)
        if args.use_wandb:
            wandb.log({"epoch_loss": epoch_loss})
        print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}')

        # 验证循环
        medsam_model.eval()
        val_iou_scores = []
        with torch.no_grad():
            for image, gt2D, boxes, _ in val_dataloader:
                image = image.to(device)
                boxes_np = boxes.detach().cpu().numpy()
                pred = medsam_model(image, boxes_np)
                pred_mask = pred.data.cpu().numpy() > 0.5  # 转换为二值掩码
                true_mask = gt2D.cpu().numpy() > 0.5

                for p_mask, t_mask in zip(pred_mask, true_mask):
                    val_iou_scores.append(iou_score(p_mask.squeeze(), t_mask.squeeze()))

        avg_val_iou = np.mean(val_iou_scores)
        print(f"Average Validation IoU: {avg_val_iou:.4f}")
        if args.use_wandb:
            wandb.log({"val_iou": avg_val_iou})

        # 保存最新模型权重
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "avg_val_iou": avg_val_iou,
        }
        torch.save(checkpoint, join(model_save_path, f"medsam_model_latest_epoch_{epoch}.pth"))

        # 如果有改进，则保存最佳模型
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            best_loss = epoch_loss
            torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))

    # for epoch in range(start_epoch, num_epochs):
    #     epoch_loss = 0
    #     for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader)):
    #         optimizer.zero_grad()
    #         boxes_np = boxes.detach().cpu().numpy()
    #         image, gt2D = image.to(device), gt2D.to(device)
    #         print("tupian:", image.shape, "biaoqian:", gt2D.shape)
    #         if args.use_amp:
    #             ## AMP
    #             with torch.autocast(device_type="cuda", dtype=torch.float16):
    #                 medsam_pred = medsam_model(image, boxes_np)
    #                 loss = seg_loss(medsam_pred, gt2D) + ce_loss(
    #                     medsam_pred, gt2D.float()
    #                 )
    #             scaler.scale(loss).backward()
    #             scaler.step(optimizer)
    #             scaler.update()
    #             optimizer.zero_grad()
    #         else:
    #             medsam_pred = medsam_model(image, boxes_np)
    #             loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
    #             loss.backward()
    #             optimizer.step()
    #             optimizer.zero_grad()
    #
    #         epoch_loss += loss.item()
    #         iter_num += 1
    #
    #     epoch_loss /= step
    #     losses.append(epoch_loss)
    #     if args.use_wandb:
    #         wandb.log({"epoch_loss": epoch_loss})
    #     print(
    #         f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
    #     )
    #
    #
    #     ## save the latest model
    #     checkpoint = {
    #         "model": medsam_model.state_dict(),
    #         "optimizer": optimizer.state_dict(),
    #         "epoch": epoch,
    #     }
    #     torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))
    #     ## save the best model
    #     if epoch_loss < best_loss:
    #         best_loss = epoch_loss
    #         checkpoint = {
    #             "model": medsam_model.state_dict(),
    #             "optimizer": optimizer.state_dict(),
    #             "epoch": epoch,
    #         }
    #         torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))
    #
    #     # %% plot loss
    #     plt.plot(losses)
    #     plt.title("Dice + Cross Entropy Loss")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Loss")
    #     plt.savefig(join(model_save_path, args.task_name + "train_loss.png"))
    #     plt.close()
    #

if __name__ == "__main__":
    main()
