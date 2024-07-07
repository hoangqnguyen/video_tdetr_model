import argparse
import os
from datetime import datetime
import json
from models import build_model
import pytorch_lightning as pl
from datasets import build_dataset
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train model")

    # Dataset parameters
    parser.add_argument(
        "--dataset", type=str, default="kovo_video", help="Dataset name"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/volleyball/2324/mp4",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--imgsz", type=int, nargs=2, default=(256, 256), help="Image size"
    )
    parser.add_argument("--n_frames", type=int, default=5, help="Number of frames")
    parser.add_argument("--epoch_size", type=int, default=4, help="Size of each epoch")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--normalize_labels", action="store_true", help="Normalize labels"
    )
    parser.add_argument(
        "--calculate_velocity", action="store_true", help="Calculate velocity"
    )
    parser.add_argument(
        "--return_frame_resolution", action="store_true", help="Return frame resolution"
    )
    parser.add_argument(
        "--return_bbox_size", action="store_true", help="Return bounding box size"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")

    # Model parameters
    parser.add_argument("--model_name", type=str, default="tdetr", help="Model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--position_embedding", type=str, default="sine", help="Position embedding type"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden dimension size"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--lr_backbone", type=float, default=1e-5, help="Learning rate for backbone"
    )
    parser.add_argument("--masks", action="store_true", help="Use masks")
    parser.add_argument(
        "--backbone", type=str, default="resnet50", help="Backbone model"
    )
    parser.add_argument(
        "--dilation", action="store_true", help="Use dilated convolutions"
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--num_queries", type=int, default=3, help="Number of queries")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--nheads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=2048,
        help="Dimension of feedforward layers",
    )
    parser.add_argument(
        "--enc_layers", type=int, default=6, help="Number of encoder layers"
    )
    parser.add_argument(
        "--dec_layers", type=int, default=6, help="Number of decoder layers"
    )
    parser.add_argument(
        "--pre_norm", action="store_true", help="Use pre-norm in transformers"
    )
    parser.add_argument(
        "--use_temporal_encodings",
        action="store_true",
        help="Use temporal encodings in transformer",
    )

    # Training parameters
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=4,
        help="Number of batches to accumulate gradients",
    )

    # * Loss coefficients
    parser.add_argument("--class_loss_coef", default=1, type=float)
    parser.add_argument("--box_loss_coef", default=5, type=float)
    return parser.parse_args()


class CustomProgressBar(RichProgressBar):
    def __init__(self, args):
        super().__init__()
        self.total_steps = args.epoch_size // 2

    def on_train_epoch_start(self, trainer, pl_module):
        self.main_progress_bar.reset(total=self.total_steps)

    def get_metrics(self, *args, **kwargs):
        # don't show the version number
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items


def main(args):
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create logging and checkpoint directories
    log_dir = os.path.join("logs", f"{args.model_name}_{timestamp}")
    checkpoint_dir = os.path.join("checkpoints", f"{args.model_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save args to log and checkpoint directories
    with open(os.path.join(checkpoint_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    train_dataset = build_dataset(args.dataset, "train", args)
    val_dataset = build_dataset(args.dataset, "val", args)
    test_dataset = build_dataset(args.dataset, "test", args)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, name=args.model_name)
    csv_logger = pl_loggers.CSVLogger(save_dir=log_dir, name=args.model_name)

    model = build_model(args)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_dir,
        filename=f"{args.model_name}_{timestamp}",
        save_top_k=1,
        mode="min",
    )
    progress_bar = CustomProgressBar(args)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.device,
        logger=[tb_logger, csv_logger],
        callbacks=[
            checkpoint_callback,
        ],
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
    )

    # Evaluate the model
    best_model = model.__class__.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        backbone=model.backbone,
        transformer=model.transformer,
        num_queries=model.num_queries,
        n_frames=model.n_frames,
        use_temporal_encodings=args.use_temporal_encodings,
    )
    best_model.evaluate(test_dataloader, args.imgsz)


if __name__ == "__main__":
    args = parse_args()
    main(args)
