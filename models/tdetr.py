import math
import torch
from torch import nn, Tensor
import pytorch_lightning as pl
import torch.nn.functional as F
from einops import rearrange, reduce
from torch.utils.data import DataLoader
from util.misc import NestedTensor
from models.transformer import build_transformer
from models.backbone import build_backbone


class TemporalEncodingSine(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TDETR(pl.LightningModule):
    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        n_frames,
        use_temporal_encodings=True,
        lr=1e-4,
        lr_backbone=1e-5,
        class_loss_coef=1,
        box_loss_coef=5,
    ):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.n_frames = n_frames
        self.num_queries = num_queries
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.class_loss_coef = class_loss_coef
        self.box_loss_coef = box_loss_coef

        if use_temporal_encodings:
            self.temp_enc = TemporalEncodingSine(1, 0.1, max_len=n_frames)
        else:
            self.temp_enc = None
        hidden_dim = transformer.d_model

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.class_embed = nn.Linear(hidden_dim, 1)
        self.location_emb = nn.Linear(hidden_dim, 2)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

    def forward(self, x: NestedTensor):
        bt = x.tensors.shape[0]
        b = bt // self.n_frames
        t = self.n_frames
        features, pos = self.backbone(x)
        src, mask = features[-1].decompose()
        src = self.input_proj(src)
        c, h, w = src.shape[1:]
        if self.temp_enc is not None:
            src = rearrange(src, "(b t) c h w -> t b (h w c)", b=b, t=t, h=h, w=w, c=c)
            src = self.temp_enc(src)
            src = rearrange(src, "t b (h w c) -> (b t) c h w", b=b, t=t, h=h, w=w, c=c)

        hs = self.transformer(src, mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.location_emb(hs).sigmoid()

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}

        return out

    def decompose(self, batch):
        frames = batch["video"]
        _class = (
            batch["class"]
            .to()
            .flatten(0, 1)
            .unsqueeze(1)
            .repeat(1, self.num_queries)
            .unsqueeze(-1)
        )
        center_xy = (
            batch["center_xy"].flatten(0, 1).unsqueeze(1).repeat(1, self.num_queries, 1)
        )
        if frames.dim() < 5:
            frames = frames.unsqueeze(0)
        frames = rearrange(frames, "b t c h w -> (b t) c h w")
        masks = torch.zeros_like(frames).mean(dim=1)
        inputs = NestedTensor(frames, masks)

        targets = {
            "center_xy": center_xy,
            "class": _class,
        }

        if "velocity_xy" in batch:
            velocity_xy = (
                batch["velocity_xy"]
                .flatten(0, 1)
                .unsqueeze(1)
                .repeat(1, self.num_queries, 1)
            )
            targets["velocity_xy"] = velocity_xy

        return inputs, targets

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
        optimizer = torch.optim.Adam(
            [
                {"params": self.backbone.parameters(), "lr": self.lr_backbone},
                {"params": self.transformer.parameters()},
                {"params": self.input_proj.parameters()},
                {"params": self.class_embed.parameters()},
                {"params": self.location_emb.parameters()},
                {"params": self.query_embed.parameters()},
            ],
            lr=self.lr,
        )
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

    @torch.no_grad()
    def compute_location_error(self, outputs, targets, imgsz):
        pred_boxes = outputs["pred_boxes"].to(self.device) * torch.tensor(
            imgsz, device=self.device
        )
        target_center_xy = targets["center_xy"].to(self.device) * torch.tensor(
            imgsz, device=self.device
        )
        distance = torch.norm(pred_boxes - target_center_xy, dim=-1)
        return distance.mean()

    def evaluate(self, dataloader, imgsz):
        self.eval()
        total_error = 0.0
        i = 1
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = self.decompose(batch)
                outputs = self(inputs.to(self.device))
                error = self.compute_location_error(outputs, targets, imgsz)
                total_error += error.item()
                i += 1
        avg_error = total_error / i
        print(f"Evaluation error: {avg_error}")
        self.train()
        return avg_error


def build_model(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = TDETR(
        backbone=backbone,
        transformer=transformer,
        num_queries=args.num_queries,
        n_frames=args.n_frames,
        use_temporal_encodings=args.use_temporal_encodings,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
    )
    return model
