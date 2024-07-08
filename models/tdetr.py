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
        optimizer="adam",
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
        self.optimizer = optimizer
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.class_loss_coef = class_loss_coef
        self.box_loss_coef = box_loss_coef

        if use_temporal_encodings:
            self.temp_enc = TemporalEncodingSine(1, 0.1, max_len=n_frames)
        else:
            self.temp_enc = None
        hidden_dim = transformer.d_model
        self.in_features_dim = backbone.num_channels * n_frames

        self.input_proj = nn.Conv2d(
            self.in_features_dim, hidden_dim, kernel_size=1, groups=n_frames
        )
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.class_embed = nn.Conv1d(hidden_dim, n_frames, kernel_size=1)
        self.location_emb = nn.Conv1d(hidden_dim, n_frames * 2, kernel_size=1)
        self._init_weights()

    def _init_conv(self, conv):
        nn.init.normal_(conv.weight, std=0.01, mean=0.0)
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def _int_embedding(self, emb):
        nn.init.xavier_uniform_(emb.weight)

    def _init_weights(self):
        self._init_conv(self.input_proj)
        self._int_embedding(self.query_embed)
        self._init_conv(self.class_embed)
        self._init_conv(self.location_emb)

    def forward(self, x: NestedTensor):
        bt = x.tensors.shape[0]
        b = bt // self.n_frames
        t = self.n_frames
        features, pos = self.backbone(x)
        src, mask = features[-1].decompose()  # src should be: (b*t, c, h, w) now

        src = rearrange(src, "(b t) c h w -> b (t c) h w", b=b)
        src = self.input_proj(src)  # src should be: (b, t*c, h, w)
        c, h, w = src.shape[1:]
        if self.temp_enc is not None:
            src = rearrange(src, "b (t c) h w -> t b (h w c)", b=b, t=t, h=h, w=w)
            src = self.temp_enc(src)
            src = rearrange(src, "t b (h w c) -> b (t c) h w", b=b, t=t, h=h, w=w)

            mask = (
                rearrange(mask, "(b t) h w -> b t h w", b=b)
                .to(torch.float32)
                .mean(dim=1)
                .to(torch.bool)
            )
            pos[-1] = rearrange(pos[-1], "(b t) c h w -> b t c h w", b=b).mean(dim=1)
            # print(f"{src.shape=}, {mask.shape=}, {pos[-1].shape=}")

        hs = self.transformer(src, mask, self.query_embed.weight, pos[-1])[0]

        hs = rearrange(hs[-1], "b q c -> b c q", b=b, q=self.num_queries)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.location_emb(hs).sigmoid()

        outputs_class = rearrange(
            outputs_class, "b (t c) q -> (b t) q c", q=self.num_queries, t=t
        )
        outputs_coord = rearrange(
            outputs_coord, "b (t c) q -> (b t) q c", q=self.num_queries, t=t
        )

        out = {"pred_logits": outputs_class, "pred_boxes": outputs_coord}
        # print(f"{hs.shape=} {out['pred_boxes'].shape=} {out['pred_logits'].shape=}")

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
            on_step=False,
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
                {"params": self.transformer.parameters(), "lr": self.lr},
                {"params": self.input_proj.parameters(), "lr": self.lr},
                {"params": self.class_embed.parameters(), "lr": self.lr},
                {"params": self.location_emb.parameters(), "lr": self.lr},,
                {"params": self.query_embed.parameters(), "lr": self.lr},
            ]
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(params)
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(params)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(params, momentum=0.9)

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
        optimizer=args.optimizer,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
    )
    return model
