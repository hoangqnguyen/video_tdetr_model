from .video_dataset_iter import build as build_video_dataset
from .video_dataset_full import build as build_video_dataset_full
from .frame_dataset import build as build_frame_dataset


def build_dataset(dataset_name, split, args):
    if dataset_name in ("kovo_video", "volleyball"):
        if args.dataset_mode == "full":
            return build_video_dataset_full(split, args)
        elif args.dataset_mode == "iter":
            return build_video_dataset(split, args)
        elif args.dataset_mode == "frame":
            return build_frame_dataset(split, args)
        return build_video_dataset(split, args)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
