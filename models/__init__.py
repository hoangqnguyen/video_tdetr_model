from .tdetr import build_model as build_tdetr_model

def build_model(args):
    if args.model_name == "tdetr":
        return build_tdetr_model(args)
    else:
        raise ValueError(f"Unknown model: {args.model_name}")