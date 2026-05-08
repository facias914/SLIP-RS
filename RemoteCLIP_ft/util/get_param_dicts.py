import json
import torch
import torch.nn as nn


def match_name_keywords(n: str, name_keywords: list):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def get_param_dict(model_without_ddp: nn.Module, args):
    param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters()]}
            # {
            #     "params": [p for n, p in model_without_ddp.named_parameters() if "transformer" in n and p.requires_grad],
            #     "lr": args.lr_transformer,
            # }
        ]

    return param_dicts