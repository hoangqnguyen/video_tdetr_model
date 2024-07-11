import os
import json
import warnings
import argparse
from datetime import datetime
from models import build_model
import pytorch_lightning as pl
from datasets import build_dataset

# from datasets.helpers import collate_fn
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="Mode (train/eval)",
        choices=["train", "eval"],
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset", type=str, default="kovo_video", help="Dataset name"
    )

    parser.add_argument(
        "--dataset_mode",
        type=str,
        default="full",
        help="Dataset mode",
        choices=["full", "iter", "frame"],
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
    parser.add_argument("--n_frames", type=int, default=4,
                        help="Number of frames")
    parser.add_argument(
        "--epoch_size", type=int, default=None, help="Size of each epoch"
    )
    parser.add_argument("--random_seed", type=int,
                        default=42, help="Random seed")

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
        "--num_workers", type=int, default=2, help="Number of workers for data loading"
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")

    # Model parameters
    parser.add_argument(
        "--model_name",
        type=str,
        default="tdetr",
        help="Model name",
        choices=["tdetr", "tdetr2", "maxvit", "maxvit2",],
    )
    parser.add_argument("--device", type=str,
                        default="cuda", help="Device to use")
    parser.add_argument(
        "--position_embedding", type=str, default="sine", help="Position embedding type"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden dimension size"
    )
    parser.add_argument("--optimizer", type=str,
                        default="adam", help="Optimizer")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--lr_backbone", type=float, default=1e-5, help="Learning rate for backbone"
    )
    parser.add_argument("--masks", action="store_true", help="Use masks")
    parser.add_argument(
        "--backbone", type=str, default="resnet18", help="Backbone model"
    )
    parser.add_argument(
        "--dilation", action="store_true", help="Use dilated convolutions"
    )
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of epochs")
    parser.add_argument("--num_queries", type=int,
                        default=3, help="Number of queries")
    parser.add_argument("--dropout", type=float,
                        default=0.1, help="Dropout rate")
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

    parser.add_argument(
        "--eval_checkpoint_path",
        type=str,
        default=None,
        help="Path to the checkpoint to evaluate",
    )

    parser.add_argument(
        "--log_subfix",
        type=str,
        default="",
        help="Subfix to add to the log directory name",
    )

    # * Loss coefficients
    parser.add_argument("--class_loss_coef", default=1, type=float)
    parser.add_argument("--box_loss_coef", default=5, type=float)
    return parser.parse_args()


def main(args):

    if args.mode == "train":
        model = build_model(args)
        # Get the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Create logging and checkpoint directories
        subfix = args.log_subfix + "_" if args.log_subfix else ""
        log_dir = os.path.join(
            "logs", f"{args.model_name}_{subfix}{args.dataset}_{timestamp}"
        )
        checkpoint_dir = os.path.join(
            "checkpoints", f"{args.model_name}_{subfix}{args.dataset}_{timestamp}"
        )
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save args to log and checkpoint directories
        with open(os.path.join(checkpoint_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

        train_dataset = build_dataset(args.dataset, "train", args)
        val_dataset = build_dataset(args.dataset, "val", args)

        if args.dataset_mode == "iter":
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                # collate_fn=collate_fn
            )
        elif args.dataset_mode in ("full", "frame"):
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
                # collate_fn=collate_fn
            )
        else:
            raise ValueError(f"Unknown dataset mode: {args.dataset_mode}")

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            # collate_fn=collate_fn
        )

        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=log_dir, name=args.model_name, sub_dir=args.dataset
        )
        # csv_logger = pl_loggers.CSVLogger(save_dir=log_dir, name=args.model_name)

        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=checkpoint_dir,
            filename=f"{args.model_name}_{args.dataset}_{timestamp}",
            save_top_k=1,
            mode="min",
        )

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator=args.device,
            logger=[
                tb_logger,
                # csv_logger
            ],
            callbacks=[
                checkpoint_callback,
                RichProgressBar(),
            ],
            accumulate_grad_batches=args.accumulate_grad_batches,
        )

        trainer.fit(
            model,
            train_dataloader,
            val_dataloader,
        )
        args.eval_checkpoint_path = checkpoint_callback.best_model_path

    elif args.mode == "eval":
        # read args from checkpoint dir
        with open(
            os.path.join(os.path.dirname(
                args.eval_checkpoint_path), "args.json"), "r"
        ) as f:
            skip_keys = [
                "mode",
                "eval_checkpoint_path",
                "dataset",
                "data_path",
                "epoch_size",
                "batch_size",
            ]
            args.__dict__.update(
                {k: v for k, v in json.load(f).items() if k not in skip_keys}
            )
        model = build_model(args)

    test_dataset = build_dataset(args.dataset, "test", args)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # collate_fn=collate_fn
    )
    # Evaluate the model
    eval(model, args.eval_checkpoint_path, test_dataloader, args)


def eval(model, checkpoint_dir, test_dataloader, args):
    best_model = model.__class__.load_from_checkpoint(
        checkpoint_dir,
        backbone=args.backbone,
        transformer=getattr(model, "transformer", None),
        num_queries=getattr(model, "num_queries", None),
        n_frames=getattr(model, "n_frames", 1),
        use_temporal_encodings=args.use_temporal_encodings,
    )
    best_model.evaluate(test_dataloader, args.imgsz)


if __name__ == "__main__":
    args = parse_args()
    main(args)
