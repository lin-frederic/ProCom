from box import Box
from pprint import pprint

config = { 
    "wandb" : False,
    "paths" : {
        "imagenet" : "/nasbrain/datasets/imagenet/images/val",
        "cub" : "/nasbrain/datasets/CUB_200_2011",
        "caltech" : "/nasbrain/datasets/caltech-101",
        "food" : "/nasbrain/datasets/food-101",
        "cifarfs" : "/nasbrain/datasets/cifar_fs",
        "fungi" : "/nasbrain/datasets/fungi",
        "flowers" : "/nasbrain/datasets/oxford_flowers",
        "pets" : "/nasbrain/datasets/oxford_pets",
        "coco" : "/nasbrain/datasets/coco_full"
    },
    "sampler" : {
        "n_ways" : None, # if None, will be set to 5 for all datasets
        
        "n_shots" : 1,
        "n_queries" : 15,
    },
    
    "batch_size" : 32,
    "n_runs" : 100,

    "top_k_masks": 2, # top k masks used for each method
    "sam_cache" : "/nasbrain/f21lin/PROCOM/cache", # path of imgs for which masks have been computed,
    "dataset": "not_specified",

    "dsm": {
        "n_eigenvectors" : 5, # number of eigenvectors to use for DSM
        "lambda_color" : 10 # as in the paper
    },

    "hierarchical": {
        "nms_thr": 0.15, # threshold for non-maximum suppression (mask)
        "area_thr": 0.05, # under this area, the mask is discarded
        "sample_per_map":5, # number of points sampled from each map
        "temperature":255*0.07 # the maps are normalized to [0,1] and then multiplied by temperature
    },

    "dezoom" : 0.1 # dezoom factor for the crop of the image
}

cfg = Box(config)

if cfg.sampler.n_ways is None:
    cfg.sampler.n_ways = {}
    for k,v in cfg.paths.items():
        cfg.sampler.n_ways[k] = 5
elif isinstance(cfg.sampler.n_ways, int):
    n_ways = cfg.sampler.n_ways
    cfg.sampler.n_ways = {k:n_ways for k in cfg.paths.keys()}


if __name__ == "__main__":
    pprint(cfg)