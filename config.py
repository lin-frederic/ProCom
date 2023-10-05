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
        "n_ways" : None,
        
        "n_shots" : 1,
        "n_queries" : 15,
    },
    
    "batch_size" : 8,
    
    
}

cfg = Box(config)

cfg.sampler.n_ways = {}
for k,v in cfg.paths.items():
    cfg.sampler.n_ways[k] = 5 


if __name__ == "__main__":
    pprint(cfg)