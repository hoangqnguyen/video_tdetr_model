import os
import glob
import json
import torch
import random
import itertools
import numpy as np
import torchvision
from torchvision.transforms import v2

torchvision.set_video_backend("video_reader")


def collate_fn(batch):
    # print(batch)
    collate_batch = {}
    for key, tensor in batch[0].items():
        if isinstance(tensor, torch.Tensor):
            collate_batch[key] = torch.stack([sample[key] for sample in batch])
        else:
            collate_batch[key] = torch.stack(
                [torch.tensor(sample[key]) for sample in batch]
            )

    return collate_batch


def frame_transform(split, args):
    if split == "train":
        frame_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(args.imgsz),
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=0.4, contrast=0.5, saturation=0.4, hue=0.3
                        ),
                        v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                        v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)),
                        # v2.RandomRotation(degrees=15),
                        # v2.RandomHorizontalFlip(p=0.5),
                        v2.RandomGrayscale(p=0.1),
                        v2.RandomPosterize(bits=4, p=0.3),
                        v2.RandomEqualize(p=0.3),
                        # v2.RandomAffine(degrees=0, translate=(0.1, 0.1))
                    ],
                    p=0.5,
                ),
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
                v2.ToDtype(torch.float32, scale=True),
                # v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    return frame_transform


def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def in_split(label_path, split_videos, dataset_name):
    if dataset_name == "volleyball":
        video_name = os.path.basename(label_path).split("_")[0]
    elif dataset_name == "kovo_video":
        video_name = os.path.basename(label_path).split("_")[1]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return video_name in map(str, split_videos)


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
        "class": labels[:, 0],
        "center_xy": labels[:, 1:3],
    }
    if calculate_velocity:
        output_labels["velocity_xy"] = labels[:, -2:]

    if return_frame_resolution:
        output_labels["frame_wh"] = labels[:, [5, 6]]

    if return_bbox_size:
        output_labels["bbox_wh"] = labels[:, [3, 4]]

    return output_labels


def get_meta(video_path):
    vid = torchvision.io.VideoReader(video_path, "video")
    metadata = vid.get_metadata()
    return metadata["video"]
