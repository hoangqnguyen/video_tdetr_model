import os
import glob
import torch
import random
import itertools
import numpy as np
import torchvision
from torchvision.transforms import v2
import json

from .helpers import get_samples, load_label, get_meta, frame_transform

torchvision.set_video_backend("video_reader")


class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        data_path,
        split,
        dataset_name,
        n_frames=5,
        epoch_size=None,
        frame_transform=None,
        video_transform=None,
        random_seed=42,
        normalize_labels=True,
        calculate_velocity=False,
        return_frame_resolution=True,
        return_bbox_size=True,
    ):
        self.samples = get_samples(data_path, split, dataset_name)
        self.normalize_labels = normalize_labels
        self.calculate_velocity = calculate_velocity
        self.return_frame_resolution = return_frame_resolution
        self.return_bbox_size = return_bbox_size

        if epoch_size is None or epoch_size < 1:
            epoch_size = len(self.samples)
        self.epoch_size = epoch_size

        self.n_frames = n_frames
        self.frame_transform = frame_transform
        self.video_transform = video_transform
        random.seed(random_seed)
        print(f"{split.title()} Dataset: {len(self.samples)} samples")

    def __iter__(self):
        for i in range(self.epoch_size):
            video_path, label_path = random.choice(self.samples)
            vid = torchvision.io.VideoReader(video_path, "video")
            metadata = vid.get_metadata()
            duration = metadata["video"]["duration"][0]
            fps = metadata["video"]["fps"][0]

            max_seek = duration - self.n_frames / fps

            start = random.uniform(0.0, max_seek)
            video_frames = []  # video frame buffer
            for frame in itertools.islice(vid.seek(start), self.n_frames):
                if self.frame_transform:
                    _frame = self.frame_transform(frame["data"])
                else:
                    _frame = frame["data"]
                video_frames.append(_frame)

            label_start_idx = int(start * fps) + 1

            data = load_label(
                label_path,
                normalize_labels=self.normalize_labels,
                calculate_velocity=self.calculate_velocity,
                start_idx=label_start_idx,
                end_idx=label_start_idx + self.n_frames,
                return_bbox_size=self.return_bbox_size,
                return_frame_resolution=self.return_frame_resolution,
            )

            # Stack it into a tensor
            video = torch.stack(video_frames, 0)
            if self.video_transform:
                video = self.video_transform(video)

            data["video"] = video

            yield data


def build(split, args):

    return VideoDataset(
        data_path=args.data_path,
        dataset_name=args.dataset,
        split=split,
        n_frames=args.n_frames,
        epoch_size=args.epoch_size,
        frame_transform=frame_transform(split, args),
        # video_transform=args.video_transform,
        random_seed=args.random_seed,
        calculate_velocity=args.calculate_velocity,
        return_frame_resolution=args.return_frame_resolution,
        return_bbox_size=args.return_bbox_size,
    )
