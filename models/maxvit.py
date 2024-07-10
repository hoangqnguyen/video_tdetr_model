import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models.feature_extraction import create_feature_extractor


def _init_linear(lin):
    nn.init.xavier_normal_(lin.weight)
    if lin.bias is not None:
        torch.nn.init.zeros_(lin.bias)


class MaxVitDetection(pl.LightningModule):
    def __init__(
        self,
        num_classes=1,
        box_dim=2,
        optimizer="adamw",
        lr=1e-4,
        lr_backbone=1e-5,
        class_loss_coef=1,
        box_loss_coef=5,
        **kwargs,
    ):
        super(MaxVitDetection, self).__init__()

        self.optimizer = optimizer
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.class_loss_coef = class_loss_coef
        self.box_loss_coef = box_loss_coef

        maxvit_t = torchvision.models.maxvit_t(
            weights="MaxVit_T_Weights.IMAGENET1K_V1"
        )
        self.backbone = create_feature_extractor(
            maxvit_t, return_nodes={"classifier.4": "features"}
        )
        out_features = self.backbone.classifier._modules["3"].out_features
        self.class_emb = nn.Linear(out_features, num_classes)
        self.bbox_emb = nn.Linear(out_features, box_dim)
        _init_linear(self.class_emb)
        _init_linear(self.bbox_emb)

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        output: class (B, T, n_classes) and bounding box(B, T, 4)
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.backbone(x)
        class_emb = self.class_emb(x["features"])
        bbox_emb = self.bbox_emb(x["features"]).sigmoid()

        outputs = {
            "pred_logits": class_emb.view(B, T, -1),
            "pred_boxes": bbox_emb.view(B, T, -1),
        }
        return outputs

    def training_step(self, batch, batch_idx):
        inputs, targets = self.decompose(batch)
        outputs = self(inputs)
        loss_dict = self.compute_loss(outputs, targets)
        self.log_dict(
            {"train_" + k: v for k, v in loss_dict.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        inputs, targets = self.decompose(batch)
        outputs = self(inputs)
        loss_dict = self.compute_loss(outputs, targets)
        self.log_dict({"val_" + k: v for k, v in loss_dict.items()})
        return loss_dict["loss"]

    def configure_optimizers(self):
        params = [
            {"params": self.backbone.parameters(), "lr": self.lr_backbone},
            {"params": self.class_emb.parameters()},
            {"params": self.bbox_emb.parameters()},
        ]
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(params, lr=self.lr)
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(params, lr=self.lr)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9)

        return optimizer

    def compute_loss(self, outputs, targets):
        # Using Cross Entropy Loss for classification and L1 Loss for bounding box coordinates
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

        target_class = targets["class"]
        target_center_xy = targets["center_xy"]

        # Cross Entropy Loss for classification
        loss_class = F.binary_cross_entropy_with_logits(pred_logits, target_class)
        # L1 Loss for bounding box coordinates
        loss_boxes = F.l1_loss(target_class * pred_boxes, target_center_xy)
        # * target_class to mask out the background class

        loss = self.class_loss_coef * loss_class + self.box_loss_coef * loss_boxes / (
            self.class_loss_coef + self.box_loss_coef
        )

        return {
            "loss": loss,
            "class_loss": loss_class,
            "box_loss": loss_boxes,
        }

    def decompose(self, batch):
        frames = batch["video"]

        _class = batch["class"].unsqueeze(-1)

        center_xy = batch["center_xy"]

        if frames.dim() < 5:
            frames = frames.unsqueeze(0)

        targets = {
            "center_xy": center_xy,
            "class": _class,
        }

        if "velocity_xy" in batch:
            targets["velocity_xy"] = batch["velocity_xy"]
        return frames, targets

    @torch.no_grad()
    def compute_location_error(self, outputs, targets, imgsz):
        pred_boxes = outputs["pred_boxes"].to(self.device) * torch.tensor(
            imgsz, device=self.device
        )
        target_center_xy = targets["center_xy"].to(self.device) * torch.tensor(
            imgsz, device=self.device
        )
        distance = torch.norm(pred_boxes - target_center_xy, dim=-1).flatten()
        return distance

    def evaluate(self, dataloader, imgsz):
        self.eval()
        all_distance = torch.empty(0)
        total_error = 0.0
        thresholds = torch.arange(11)
        count = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = self.decompose(batch)
                outputs = self(inputs.to(self.device))
                distance = self.compute_location_error(outputs, targets, imgsz).cpu()
                all_distance = torch.cat((all_distance, distance))
                total_error += distance.sum().item()
                count += distance.numel()
        avg_error = total_error / count
        for th in thresholds:
            acc = (all_distance <= th).sum() / count
            print(f"Accuracy @ {th}: {acc:.2%}")
        print(f"Mean distance error: {avg_error}")
        self.train()
        return avg_error


def build_model(args):
    model = MaxVitDetection(
        num_classes=1,
        box_dim=2,
        backbone=args.backbone,
        optimizer=args.optimizer,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
    )
    return model