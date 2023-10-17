import torch
from torch import nn


load_refs = {
    "s":"dinov2_vits14",
    "b":"dinov2_vitb14",
    "l":"dinov2_vitl14",
    "g":"dinov2_vitg14"
}

repo_ref = "facebookresearch/dinov2"


def get_model(size="s",use_v2=False):
    if use_v2:
        if size == "s":
            model = torch.hub.load(repo_ref, load_refs[size])
        elif size == "b":
            model = torch.hub.load(repo_ref, load_refs[size])
        elif size == "l":
            model = torch.hub.load(repo_ref, load_refs[size])
        elif size == "g":
            model = torch.hub.load(repo_ref, load_refs[size])
    else:
        if size == "s":
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        elif size == "b":
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        elif size == "l":
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vitl16')
    return model

# adapted from the official repos
def forward_dino_v1(model, x):
    x = model.prepare_tokens(x)
    for blk in model.blocks:
        x = blk(x)
    x = model.norm(x)
    return x

# for dinov2, pass the is_training = True flag