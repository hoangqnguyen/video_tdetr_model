from .kovo_video import build as build_kovo_video

def build_dataset(dataset_name, split, args):
    if dataset_name == "kovo_video":
        return build_kovo_video(split, args)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")