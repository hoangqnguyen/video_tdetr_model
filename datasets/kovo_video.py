import os
import glob
import torch
import random
import itertools
import numpy as np
import torchvision
from torchvision.transforms import v2
import json

torchvision.set_video_backend("video_reader")


def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def in_split(label_path, split_videos):
    video_name = os.path.basename(label_path).split("_")[1]
    # print (split_videos)
    return video_name in map(str, split_videos)


def get_samples(data_path, split):
    split_dict = load_json(os.path.join(data_path, "split.json"))
    split_videos = split_dict[split]
    labels = sorted(
        [
            label_path
            for label_path in glob.glob(os.path.join(data_path, "*.txt"))
            if in_split(label_path, split_videos)
        ]
    )
    videos = [label.replace(".txt", ".mp4") for label in labels]
    return list(zip(videos, labels))


def load_label(
    label_path,
    start_idx=None,
    end_idx=None,
    normalize_labels=False,
    calculate_velocity=False,
    return_frame_resolution=False,
    return_bbox_size=False,
):
    labels = np.loadtxt(label_path)

    start_idx = start_idx or 0
    end_idx = end_idx or labels.shape[0]
    n_frames = end_idx - start_idx

    labels = labels[start_idx:end_idx]

    # add first column +1 at non-nan rows, coz 0 is reserved for non-object (~ non-ball)
    non_nan_rows = ~np.isnan(labels[:, 0])
    labels[non_nan_rows, 0] += 1
    labels = labels.astype(float)

    if normalize_labels:
        # normalize the labels to [0, 1]
        labels[non_nan_rows, 1] /= labels[non_nan_rows, 5]
        labels[non_nan_rows, 2] /= labels[non_nan_rows, 6]
        labels[non_nan_rows, 3] /= labels[non_nan_rows, 5]
        labels[non_nan_rows, 4] /= labels[non_nan_rows, 6]

    if calculate_velocity:
        # add two new cols for velocity x and y at the end of labels
        labels = np.concatenate([labels, np.zeros((labels.shape[0], 2))], axis=1)

        # extend mask because if a row is nan, then we can't calculate velocity the velocity at that row and the row before it
        mask_velocity = non_nan_rows & np.roll(non_nan_rows, 1)

        labels_vel = labels[mask_velocity]

        velocity = labels_vel[1:, [1, 2]] - labels_vel[:-1, [1, 2]]

        # concate zeros at the beginning of velocity
        velocity = np.concatenate([np.zeros((1, 2)), velocity], axis=0)
        labels[mask_velocity, -2:] = velocity

        if normalize_labels:
            # normalize the velocity
            labels[mask_velocity, -2] /= labels[mask_velocity, 5]
            labels[mask_velocity, -1] /= labels[mask_velocity, 6]

    # fill nan with zeros
    labels = np.nan_to_num(labels, nan=0)

    output_labels = {
        "has_ball": labels[:, 0],
        "center_xy": labels[:, 1:3],
    }
    if calculate_velocity:
        output_labels["velocity_xy"] = labels[:, -2:]

    if return_frame_resolution:
        output_labels["frame_wh"] = labels[:, [5, 6]]

    if return_bbox_size:
        output_labels["bbox_wh"] = labels[:, [3, 4]]

    output_labels["class"] = labels[:, 0]

    return output_labels


class KOVOVideoDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        data_path,
        split,
        n_frames=5,
        epoch_size=None,
        frame_transform=None,
        video_transform=None,
        random_seed=42,
        normalize_labels=False,
        calculate_velocity=False,
        return_frame_resolution=True,
        return_bbox_size=True,
    ):
        self.samples = get_samples(data_path, split)
        self.normalize_labels = normalize_labels
        self.calculate_velocity = calculate_velocity
        self.return_frame_resolution = return_frame_resolution
        self.return_bbox_size = return_bbox_size

        if epoch_size is None:
            epoch_size = len(self.samples)
        self.epoch_size = epoch_size

        self.n_frames = n_frames
        self.frame_transform = frame_transform
        self.video_transform = video_transform
        random.seed(random_seed)
        print(f"Dataset with {len(self.samples)} samples")

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


def frame_transform(split, args):

    if split == "train":
        frame_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(args.imgsz),
                # v2.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.4, hue=0.3),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    elif split in ["val", "test"]:
        frame_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(args.imgsz),
                # v2.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.4, hue=0.3),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    return frame_transform


def build(split, args):

    return KOVOVideoDataset(
        data_path=args.data_path,
        split=split,
        n_frames=args.n_frames,
        epoch_size=args.epoch_size,
        frame_transform=frame_transform(split, args),
        # video_transform=args.video_transform,
        random_seed=args.random_seed,
        normalize_labels=args.normalize_labels,
        calculate_velocity=args.calculate_velocity,
        return_frame_resolution=args.return_frame_resolution,
        return_bbox_size=args.return_bbox_size,
    )
