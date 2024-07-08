from .tdetr import build_model as build_tdetr_model
from .tdetr2 import build_model as build_tdetr_model2

def build_model(args):
    if args.model_name == "tdetr":
        return build_tdetr_model(args)
    elif args.model_name == "tdetr2":
        return build_tdetr_model2(args)
    else:
        raise ValueError(f"Unknown model: {args.model_name}")