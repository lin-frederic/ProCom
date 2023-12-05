from box import Box
from pprint import pprint

config = { 
    "paths" : {
        "imagenet" : "/nasbrain/datasets/imagenet/images/val",
        "cub" : "/nasbrain/datasets/CUB_200_2011",
        "caltech" : "/nasbrain/datasets/caltech-101",
        "food" : "/nasbrain/datasets/food-101",
        "cifarfs" : "/nasbrain/datasets/cifar_fs",
        "fungi" : "/nasbrain/datasets/fungi",
        "flowers" : "/nasbrain/datasets/oxford_flowers",
        "pets" : "/nasbrain/datasets/oxford_pets",
    },
    "sampler" : {
        "n_ways" : None, # if None, will be set to 5 for all datasets
        
        "n_shots" : 1,
        "n_queries" : 15,
    },
    
    "batch_size" : 32,
    "n_runs" : 10,

    "top_k": 1, # top k masks used for each method
    
    
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