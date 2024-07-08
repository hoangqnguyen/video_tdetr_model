from .video_dataset import build as build_video_dataset


def build_dataset(dataset_name, split, args):
    if dataset_name in ("kovo_video", "volleyball"):
        return build_video_dataset(split, args)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
