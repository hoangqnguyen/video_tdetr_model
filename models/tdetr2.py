import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


def _init_conv(conv):
    nn.init.normal_(conv.weight, std=0.01, mean=0.0)
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def _init_linear(lin):
    nn.init.xavier_normal_(lin.weight)
    if lin.bias is not None:
        torch.nn.init.zeros_(lin.bias)

def _init_embedding(emb):
    nn.init.xavier_uniform_(emb.weight)


class TemporalPositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(TemporalPositionEncoding, self).__init__()
        self.d_model = d_model

        # Create a long enough P matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class TemporalBackbone(nn.Module):
    def __init__(self, backbone="resnet18", out_dim=256):
        super(TemporalBackbone, self).__init__()
        resnet = getattr(torchvision.models, backbone)(pretrained=True)

        # Use 3D convolutions for the initial layer to handle the temporal dimension
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)  # Use 3D batch normalization
        self.relu = resnet.relu

        # 2D Convolutions and rest of the ResNet layers
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.avgpool = resnet.avgpool
        hidden_dim = resnet.fc.in_features
        self.fc = nn.Linear(hidden_dim, out_dim)
        _init_conv(self.conv1)
        _init_linear(self.fc)

    def forward(self, x):
        # Input x has shape (B, T, C, H, W), we need to permute it to (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv1(x)
        x = self.bn1(x)  # Use 3D batch normalization
        x = self.relu(x)

        # Flatten temporal dimension and use as batch dimension for 2D convolutions
        B, C, T, H, W = x.shape
        x = x.view(B * T, C, H, W)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        # Reshape back to (B, T, -1) if needed
        x = x.view(B, T, -1)
        return x


class MultiFrameTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super(MultiFrameTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead

        # Encoder and Decoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)

        # Temporal Position Encoding
        self.temporal_pos_encoder = TemporalPositionEncoding(d_model)

        # Output projection
        self.linear = nn.Linear(d_model, d_model)
        _init_linear(self.linear)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # Add temporal position encoding to source and target
        src = self.temporal_pos_encoder(src)
        tgt = self.temporal_pos_encoder(tgt)

        # Pass through the transformer
        memory = self.encoder(src, mask=src_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        # Final linear projection
        output = self.linear(output)
        return output


class DetectionHead(nn.Module):
    def __init__(self, d_model, num_classes, box_dim):
        super(DetectionHead, self).__init__()
        self.class_embed = nn.Linear(d_model, num_classes)
        self.bbox_embed = nn.Linear(d_model, box_dim)
        _init_linear(self.class_embed)
        _init_linear(self.bbox_embed)

    def forward(self, x):
        classes = self.class_embed(x)
        boxes = self.bbox_embed(x).sigmoid()
        return classes, boxes


class VideoObjectDetection(pl.LightningModule):
    def __init__(
        self,
        backbone="resnet18",
        d_model=256,
        num_classes=1,
        box_dim=2,
        num_queries=3, n_frames=4,
        optimizer="adam",
        lr=1e-4,
        lr_backbone=1e-5,
        class_loss_coef=1,
        box_loss_coef=5,
        **kwargs,
    ):
        super(VideoObjectDetection, self).__init__()

        self.d_model = d_model
        self.num_queries = num_queries
        self.n_frames = n_frames
        
        self.optimizer = optimizer
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.class_loss_coef = class_loss_coef
        self.box_loss_coef = box_loss_coef
        if type(backbone) == str:
            self.backbone = TemporalBackbone(backbone=backbone, out_dim=d_model)
        elif isinstance(backbone, TemporalBackbone):
            self.backbone = backbone
        self.transformer = MultiFrameTransformer(d_model=d_model)
        self.detection_head = DetectionHead(
            d_model=d_model, num_classes=num_classes, box_dim=box_dim
        )

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.temporal_pos_encoder = TemporalPositionEncoding(d_model)

        _init_embedding(self.query_embed)


    def forward(self, x):
        features = self.backbone(x)
        B, T, C = features.shape

        # Prepare src and tgt for the transformer
        # src = features.permute(1, 0, 2)  # (T, B, d_model)
        src = self.temporal_pos_encoder(features.permute(1, 0, 2))  # (T, B, d_model)

        # tgt = torch.zeros_like(src)  # Example target, can be adjusted as needed
       
        # Prepare query embeddings for each frame
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # (num_queries, B, d_model)
        query_embed = query_embed.unsqueeze(0).repeat(T, 1, 1, 1)  # (T, num_queries, B, d_model)
        query_embed = query_embed.view(T * self.num_queries, B, -1)  # (T * num_queries, B, d_model)

        # Pass through the transformer
        transformer_output = self.transformer(src, query_embed)  # (T * num_queries, B, d_model)

        # Reshape transformer output to (B, T, num_queries, d_model)
        transformer_output = transformer_output.view(T, self.num_queries, B, -1).permute(2, 0, 1, 3)  # (B, T, num_queries, d_model)

        # Pass through the detection head
        classes, boxes = self.detection_head(transformer_output)  # (B, T, num_queries, num_classes), (B, T, num_queries, box_dim)
        out = {"pred_logits": classes, "pred_boxes": boxes}
        return out

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
            {"params": self.transformer.parameters()},
            {"params": self.detection_head.parameters()},
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
        
        _class = (
            batch["class"] # (b, t)
            # .flatten(0, 1)
            .unsqueeze(2)
            .repeat(1, 1, self.num_queries)
            .unsqueeze(-1)
        )
        
        center_xy = (
            batch["center_xy"]
            # .flatten(0, 1)
            .unsqueeze(
                2).repeat(1,1, self.num_queries, 1)
        )

        if frames.dim() < 5:
            frames = frames.unsqueeze(0)

        targets = {
            "center_xy": center_xy,
            "class": _class,
        }

        if "velocity_xy" in batch:
            targets["velocity_xy"] = (
                batch["velocity_xy"]
                # .flatten(0, 1)
                .unsqueeze(2)
                .repeat(2, self.num_queries, 1)
            )

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
    model = VideoObjectDetection(
        num_classes=1,
        box_dim=2,
        backbone=args.backbone,
        d_model=args.hidden_dim,
        optimizer=args.optimizer,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
    )
    return model
