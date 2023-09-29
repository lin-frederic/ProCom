from box import Box

config = { 
    "paths" : {
        "imagenet" : "/nasbrain/datasets/imagenet/images/val",
        "tieredimagenet" : "/nasbrain/datasets/tieredimagenet",
        "cub" : "/nasbrain/datasets/CUB_200_2011",
        "dtd" : "/nasbrain/datasets/dtd",
        "aircraft" : "/nasbrain/datasets/fgvc-aircraft-2013b",
        "caltech" : "/nasbrain/datasets/caltech-101",
        "food" : "/nasbrain/datasets/food-101",
    }
}

cfg = Box(config)