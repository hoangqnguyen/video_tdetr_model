from .video_dataset_iter import build as build_video_dataset
from .video_dataset_full import build as build_video_dataset_full

def build_dataset(dataset_name, split, args):
    if dataset_name in ("kovo_video", "volleyball"):
        if args.dataset_mode == "full":
            return build_video_dataset_full(split, args)
        return build_video_dataset(split, args)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
