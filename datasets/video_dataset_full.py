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


class VideoDataset(torch.utils.data.Dataset):
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
        self.samples = get_samples(data_path, split, dataset_name)
        self.normalize_labels = normalize_labels
        self.calculate_velocity = calculate_velocity
        self.return_frame_resolution = return_frame_resolution
        self.return_bbox_size = return_bbox_size

        self.n_frames = n_frames
        self.frame_transform = frame_transform
        self.video_transform = video_transform
        random.seed(random_seed)
        print(f"{split.title()} Dataset: {len(self.samples)} samples")

        self.total_frames = sum(
            int(get_meta(video_path)["duration"][0] * get_meta(video_path)["fps"][0])
            for video_path, _ in self.samples
        )

        self.video_cache = {}  # Cache to store loaded videos

    def __len__(self):
        return self.total_frames - len(self.samples) * (self.n_frames - 1)

    def _load_video(self, video_path):
        """Helper function to load video into memory."""
        if video_path in self.video_cache:
            return self.video_cache[video_path]

        vid = torchvision.io.VideoReader(video_path, "video")
        self.video_cache[video_path] = vid
        return vid

    def __getitem__(self, idx):
        for video_path, label_path in self.samples:
            vid = self._load_video(video_path)
            # vid = torchvision.io.VideoReader(video_path, "video")
            metadata = vid.get_metadata()
            duration = metadata["video"]["duration"][0]
            fps = metadata["video"]["fps"][0]

            n_total_frames = int(duration * fps)

            if idx < n_total_frames - (self.n_frames - 1):
                start_frame = idx
                video_frames = []  # video frame buffer
                vid.seek(start_frame / fps)
                for frame in itertools.islice(vid, self.n_frames):
                    if self.frame_transform:
                        _frame = self.frame_transform(frame["data"])
                    else:
                        _frame = frame["data"]
                    video_frames.append(_frame)

                # If we have less than n_frames, pad the sequence with zeros
                if len(video_frames) < self.n_frames:
                    padding_frames = [
                        torch.zeros_like(video_frames[0])
                        for _ in range(self.n_frames - len(video_frames))
                    ]
                    video_frames.extend(padding_frames)

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

            idx -= n_total_frames - (self.n_frames - 1)

        raise IndexError("Index out of range")


def build(split, args):

    return VideoDataset(
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
