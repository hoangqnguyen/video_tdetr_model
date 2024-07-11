import math
import torch
import torchvision
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import pytorch_lightning as pl
from x_transformers import Decoder
from .maxvit_core import maxvit_tiny_tf_512 as core
from torchvision.models.feature_extraction import create_feature_extractor
from positional_encodings.torch_encodings import PositionalEncoding2D, PositionalEncoding3D, Summer


def _init_linear(lin):
    nn.init.xavier_normal_(lin.weight)
    if lin.bias is not None:
        torch.nn.init.zeros_(lin.bias)


def _init_embedding(emb):
    nn.init.xavier_uniform_(emb.weight)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

        for i, layer in enumerate(self.layers):
            _init_linear(layer)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MaxViT_Encoder(nn.Module):
    def __init__(self, stack_immidiate_outputs=False):
        super().__init__()

        self.backbone = core(pretrained=True)

        self.out_dim = 960 if stack_immidiate_outputs else 512
        self.stack_immdiate_outputs = stack_immidiate_outputs
        if self.stack_immdiate_outputs:
            self.max_pool0 = nn.MaxPool2d(8)
            self.max_pool1 = nn.MaxPool2d(4)
            self.max_pool2 = nn.MaxPool2d(2)

    def forward(self, x):
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output
            return hook

        for i in range(4):
            self.backbone.stages[i].register_forward_hook(
                get_activation(f'stages.{i}'))

        _ = self.backbone(x)

        if not self.stack_immdiate_outputs:
            return activation["stages.3"]

        x0 = self.max_pool0(activation["stages.0"])
        x1 = self.max_pool1(activation["stages.1"])
        x2 = self.max_pool2(activation["stages.2"])
        x3 = activation["stages.3"]

        out = torch.cat([x0, x1, x2, x3], dim=1)
        return out


class MaxVit_Detection2(pl.LightningModule):
    def __init__(
        self,
        num_classes=1,
        box_dim=2,
        optimizer="adamw",
        lr=1e-4,
        lr_backbone=1e-5,
        class_loss_coef=1,
        box_loss_coef=5,
        num_queries=1,
        multiscale_features=True,
        decoder_depth=6,
        decoder_heads=8,
        n_frames=1,
        use_nested_tensors=False,
        ** kwargs,
    ):
        super(MaxVit_Detection2, self).__init__()

        self.num_classes = num_classes
        self.box_dim = box_dim
        self.num_queries = num_queries
        self.multiscale_features = multiscale_features
        self.use_nested_tensors = use_nested_tensors
        self.multiframes = n_frames > 1

        self.optimizer = optimizer
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.class_loss_coef = class_loss_coef
        self.box_loss_coef = box_loss_coef

        self.backbone = MaxViT_Encoder(stack_immidiate_outputs=True)
        hidden_dim = self.backbone.out_dim

        self.decoder = Decoder(
            dim=hidden_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            cross_attend=True,
            attn_flash=True,  # just set this to True if you have pytorch 2.0 installed
        )
        if self.multiframes:
            self.pos_encoder = Summer(PositionalEncoding3D(hidden_dim))
        else:
            self.pos_encoder = Summer(PositionalEncoding2D(hidden_dim))

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, box_dim, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # q, c

        _init_linear(self.class_embed)
        _init_embedding(self.query_embed)

    def forward(self, samples):
        if self.use_nested_tensors:
            src = samples.tensors
        else:
            src = samples

        assert src.ndim in (4, 5), f"Expected 4D or 5D input (got {src.ndim}D)"

        if src.ndim == 5:
            bs, t, c, h, w = src.shape
            src = src.flatten(0, 1)  # bs*t, c, h, w
            context = self.backbone(src)  # b*t, c, h, w
            context = rearrange(context, "(b t) c h w -> b t h w c", b=bs, t=t)
            context_with_pos = self.pos_encoder(
                context).flatten(1, 3)  # b, t*h*w, c
            object_queries = self.query_embed.weight.unsqueeze(
                0).repeat(bs, t, 1)  # (b, t*q, c)

        elif src.ndim == 4:
            bs, c, h, w = src.shape
            context = self.backbone(src).permute(0, 2, 3, 1)  # b, h, w, c
            context_with_pos = self.pos_encoder(
                context).flatten(1, 2)  # b, h*w, c
            object_queries = self.query_embed.weight.unsqueeze(
                0).repeat(bs, 1, 1)  # b, q, c

        # print(f"context_with_pos.shape: {context_with_pos.shape}")
        # print(f"object_queries.shape: {object_queries.shape}")

        hs = self.decoder(object_queries, context=context_with_pos)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        if self.multiframes:
            outputs_class = outputs_class.view(bs, t, -1, self.num_classes + 1)
            outputs_coord = outputs_coord.view(bs, t, -1, self.box_dim)

        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}

        return out

    def common_step(self, batch, batch_idx):
        inputs, targets = self.decompose(batch)
        outputs = self(inputs)
        loss_dict = self.compute_loss(outputs, targets)
        return loss_dict

    def training_step(self, batch, batch_idx):
        loss_dict = self.common_step(batch, batch_idx)
        self.log_dict(
            {"train_" + k: v for k, v in loss_dict.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict = self.common_step(batch, batch_idx)
        self.log_dict({"val_" + k: v for k, v in loss_dict.items()})
        return loss_dict["loss"]

    def configure_optimizers(self):
        params = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad],
            },
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]

        optimizers = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
        }
        optimizer_class = optimizers.get(self.optimizer)
        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        optimizer = optimizer_class(params, lr=self.lr)
        return optimizer

    def compute_loss(self, outputs, targets):
        # Using Cross Entropy Loss for classification and L1 Loss for bounding box coordinates
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

        target_class = targets["class"]
        target_center_xy = targets["center_xy"]

        # print(
        #     f"{pred_logits.shape=} {pred_boxes.shape=} {target_class.shape=} {target_center_xy.shape=}")

        # Cross Entropy Loss for classification
        # loss_class = F.binary_cross_entropy_with_logits(pred_logits, target_class)
        loss_class = F.cross_entropy(
            pred_logits.flatten(0, -2), target_class.flatten())
        # L1 Loss for bounding box coordinates
        loss_boxes = F.l1_loss(
            target_class.unsqueeze(-1) * pred_boxes, target_center_xy
        )
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

        _class = batch["class"].to(
            torch.long).unsqueeze(-1).repeat(1, 1, self.num_queries)

        center_xy = batch["center_xy"].unsqueeze(
            2).repeat(1, 1, self.num_queries, 1)

        if frames.dim() < 5:
            frames = frames.unsqueeze(0)

        targets = {
            "center_xy": center_xy,
            "class": _class,
        }

        if "velocity_xy" in batch:
            targets["velocity_xy"] = batch["velocity_xy"].repeat(
                1, 1, self.num_queries, 1)
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
                distance = self.compute_location_error(
                    outputs, targets, imgsz).cpu()
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
    model = MaxVit_Detection2(
        num_classes=1,
        box_dim=2,
        backbone=args.backbone,
        optimizer=args.optimizer,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        decoder_depth=args.dec_layers,
        decoder_heads=args.nheads,
        num_queries=args.num_queries,
        n_frames=args.n_frames,
    )
    return model
