import os
import random

import numpy as np
import torch
import yaml
from addict import Dict
from pytorch_lightning import Callback, LightningModule
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torchsummary import summary

from loss import binary_dice_coefficient, binary_mean_iou, symmetric_lovasz
from transform import train_transform_K, valid_transform_K
from utils import get_args, object_from_dict
from transformers import SegformerForSemanticSegmentation


def get_model(cfg):
    return object_from_dict(cfg.model)


class SegFormer(nn.Module):
    def __init__(
        self, pretrained: str = "nvidia/segformer-b4-finetuned-ade-512-512", size: int = 512, num_labels: int = 9
    ):
        super().__init__()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            pretrained, ignore_mismatched_sizes=True, num_labels=num_labels
        )
        self.size = size

    def forward(self, x):
        outputs = self.segformer(x)

        upsampled_logits = torch.nn.functional.interpolate(
            outputs.logits, size=(self.size, self.size), mode="bilinear", align_corners=False
        )
        return upsampled_logits


class FineTuneCB(Callback):
    # add callback to freeze/unfreeze trained layers
    def __init__(self, unfreeze_epoch: int) -> None:
        self.unfreeze_epoch = unfreeze_epoch

    def on_epoch_start(self, trainer, pl_module) -> None:
        if trainer.current_epoch != self.unfreeze_epoch:
            return
        for n, param in pl_module.model.named_parameters():
            param.requires_grad = True
        optimizers, _ = pl_module.configure_optimizers()
        trainer.optimizers = optimizers


class LitNailSegmentation(LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float = 1e-4):
        super().__init__()
        self.model = model
        self.loss_function = symmetric_lovasz
        self.dice_soft = binary_dice_coefficient
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, masks = batch["image"], batch["label"]

        imgs, masks = imgs.float(), masks.float()
        logits = self(imgs)

        train_loss = self.loss_function(logits, masks)
        train_dice_soft = self.dice_soft(logits, masks)

        self.log("train_loss", train_loss, prog_bar=True)
        self.log("train_dice_soft", train_dice_soft, prog_bar=True)
        return {"loss": train_loss, "train_dice_soft": train_dice_soft}

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch["image"], batch["label"]

        imgs, masks = imgs.float(), masks.float()
        logits = self(imgs)
        valid_loss = self.loss_function(logits, masks)
        valid_dice_soft = self.dice_soft(logits, masks)
        valid_iou = binary_mean_iou(logits, masks)

        self.log("valid_loss", valid_loss, prog_bar=True)
        self.log("valid_dice", valid_dice_soft, prog_bar=True)
        self.log("valid_iou", valid_iou, prog_bar=True)

        return {
            "valid_loss": valid_loss,
            "valid_dice": valid_dice_soft,
            "valid_iou": valid_iou,
        }

    def validation_epoch_end(self, outputs):

        logs = {"epoch": self.trainer.current_epoch}
        valid_losses = torch.stack([x["valid_loss"] for x in outputs]).mean()
        valid_dices = torch.stack([x["valid_dice"] for x in outputs]).mean()
        valid_ious = torch.stack([x["valid_iou"] for x in outputs]).mean()

        logs["valid_losses"] = valid_losses
        logs["valid_dices"] = valid_dices
        logs["valid_ious"] = valid_ious

        return {
            "valid_losses": valid_losses,
            "valid_dices": valid_dices,
            "valid_ious": valid_ious,
            "log": logs,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, 0)
        self.optimizer = [optimizer]
        return self.optimizer, [scheduler]


if __name__ == "__main__":
    segformer = SegFormer("nvidia/segformer-b3-finetuned-ade-512-512", 512, 1)
    inputs = torch.rand((16, 3, 512, 512))
    outputs = segformer(inputs)
    print(outputs.shape)
