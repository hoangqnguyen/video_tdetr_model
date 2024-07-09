import os
import glob
import torch
import random
import itertools
import numpy as np
import torchvision
from torchvision.transforms import v2
import json
from natsort import natsorted

from .helpers import load_label, frame_transform, load_json, in_split


def get_frames(data_path, label_path, dataset_name):
    basename = os.path.basename(label_path)
    if dataset_name == "volleyball":
        dir1 = basename.split("_")[0]
        dir2 = basename.split("_")[1].split(".")[0]
        frames = glob.glob(os.path.join(data_path, "frames", dir1, dir2, "*jpg"))
    elif dataset_name == "kovo_video":
        dir1 = basename.split("_")[1:3]
        dir2 = "_".join(basename.replace(".txt", "").split("_")[-2:])
        frames = glob.glob(os.path.join(data_path, "frames", *dir1, dir2, "*jpg"))
    return natsorted(frames)


def get_samples(data_path, split, dataset_name):
    split_dict = load_json(os.path.join(data_path, "split.json"))
    split_videos = split_dict[split]

    labels = sorted(
        [
            label_path
            for label_path in glob.glob(os.path.join(data_path, "*.txt"))
            if in_split(label_path, split_videos, dataset_name)
        ]
    )

    data = [
        {"label": label_path, "frames": get_frames(data_path, label_path, dataset_name)}
        for label_path in labels
    ]

    return data


class FrameDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        split,
        dataset_name,
        n_frames=5,
        frame_transform=None,
        video_transform=None,
        random_seed=42,
        normalize_labels=True,
        calculate_velocity=False,
        return_frame_resolution=True,
        return_bbox_size=True,
    ):
        self.data_path = data_path
        self.split = split
        self.dataset_name = dataset_name
        self.normalize_labels = normalize_labels
        self.calculate_velocity = calculate_velocity
        self.return_frame_resolution = return_frame_resolution
        self.return_bbox_size = return_bbox_size

        self.n_frames = n_frames
        self.frame_transform = frame_transform
        self.video_transform = video_transform
        random.seed(random_seed)

        self.samples = get_samples(data_path, split, dataset_name)
        print(f"{split.title()} Dataset: {len(self.samples)} samples")

        self.total_frames = sum(len(sample["frames"]) for sample in self.samples)

    def __len__(self):
        return self.total_frames - len(self.samples) * (self.n_frames - 1)

    def __getitem__(self, idx):
        cumulative_count = 0
        for sample in self.samples:
            frame_files = sample["frames"]
            label_path = sample["label"]
            n_total_frames = len(frame_files)

            if (
                cumulative_count
                <= idx
                < cumulative_count + n_total_frames - (self.n_frames - 1)
            ):
                start_frame = idx - cumulative_count
                video_frames = []  # frame buffer

                try:
                    for frame_idx in range(start_frame, start_frame + self.n_frames):
                        if frame_idx < n_total_frames:
                            frame = torchvision.io.read_image(frame_files[frame_idx])
                            if self.frame_transform:
                                frame = self.frame_transform(frame)
                            video_frames.append(frame)
                        else:
                            padding_frame = torch.zeros_like(video_frames[0])
                            video_frames.append(padding_frame)

                    video_frames = video_frames[: self.n_frames]  # Ensure exact length
                    video = torch.stack(video_frames, 0)

                    label_start_idx = start_frame

                    data = load_label(
                        label_path,
                        normalize_labels=self.normalize_labels,
                        calculate_velocity=self.calculate_velocity,
                        start_idx=label_start_idx,
                        end_idx=label_start_idx + self.n_frames,
                        return_bbox_size=self.return_bbox_size,
                        return_frame_resolution=self.return_frame_resolution,
                    )

                    if self.video_transform:
                        video = self.video_transform(video)

                    data["video"] = video

                    return data

                except Exception as e:
                    print(
                        f"Error processing frames in {label_path} at frame {start_frame}: {e}"
                    )
                    raise e

            cumulative_count += n_total_frames - (self.n_frames - 1)

        raise IndexError("Index out of range")


def build(split, args):
    return FrameDataset(
        data_path=args.data_path,
        dataset_name=args.dataset,
        split=split,
        n_frames=args.n_frames,
        frame_transform=frame_transform(split, args),
        # video_transform=args.video_transform,
        random_seed=args.random_seed,
        calculate_velocity=args.calculate_velocity,
        return_frame_resolution=args.return_frame_resolution,
        return_bbox_size=args.return_bbox_size,
    )
