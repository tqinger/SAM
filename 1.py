
import os

join = os.path.join

import torch
import torch.nn as nn
from mobile_sam import msam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from segment_anything import sam_model_registry
import torch.nn.functional as F


class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        img_en,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.img_en = img_en
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        img_em = self.img_en(image)
        image_embedding = image_embedding*0.5 + img_em*0.5
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


if __name__ == "__main__" :

    sam_model = sam_model_registry["vit_b"](checkpoint=r"C:\Users\Public\cv\sam\playground\label_anything\sam_vit_b.pth")
    model_type = "vit_t"
    sam_checkpoint = r"C:\Users\tdqin\Desktop\MobileSAM-master\MobileSAM-master\weights\mobile_sam.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mobile_sam = msam_model_registry[model_type](checkpoint=sam_checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        img_en=mobile_sam.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to("cpu")
    # print(medsam_model)
    total_params = sum(p.numel() for p in sam_model.image_encoder.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")

    # 如果您还想得到非训练参数的数量（例如，冻结的参数）
    total_non_trainable_params = sum(p.numel() for p in sam_model.image_encoder.parameters() if not p.requires_grad)
    print(f"Total number of non-trainable parameters: {total_non_trainable_params}")

    # 打印总参数量
    print(f"Total number of parameters: {total_params + total_non_trainable_params}")