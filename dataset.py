""""
Mini-imagenet dataset
 CIFAR-FS dataset
CUB dataset
DTD dataset (wtf is that)
Omniglot dataset
Fungi dataset
FGVC Aircraft dataset
MSCOCO dataset
caltech101 dataset
food101 dataset
"""

import os
import random as rd
from typing import Any
from config import cfg



class FolderExplorer():
    def __init__(self, dataset_paths) -> None:
        self.dataset_paths = dataset_paths
    
    def __call__(self) -> Any:
        # returns a dict where the key is the dataset name and the value is a list of list (first index is the class index)
        
        # initialize the dict
        dataset_dict = {}

        for dataset in self.dataset_paths:
            if dataset=="imagenet":
                dataset_dict["imagenet"]={}
                for class_name in os.listdir(dataset):
                    dataset_dict["imagenet"][class_name] = [os.path.join(dataset, class_name, image_name) for image_name in os.listdir(os.path.join(dataset, class_name))]
                    

            elif dataset=="cub":
                with open(os.path.join(cfg.dataset_path, "train_test_split.txt")) as f:
                    train_test_split = f.readlines()
                    # filter train dataset with the image ids
                    train_dataset_image_ids = [line.split()[0] for line in train_test_split if line.split()[1]=="1"]
                    
                with open(os.path.join(cfg.dataset_path, "images.txt")) as f:
                    images = f.readlines()
                    # filter the train dataset with the image ids and class ids
                    train_dataset_images_ids_class = { line.split()[0]: line.split()[1] for line in images if line.split()[0] in train_dataset_image_ids}

                dataset_dict["cub"] = train_dataset_image_ids

            elif dataset=="cifarfs":
                pass
            elif dataset=="omniglot":
                pass
            elif dataset=="fungi":
                
                pass
            elif dataset=="fgvc":
                dataset_dict["fgvc"] = {}
                seen_classes = []
            elif dataset=="mscoco":
                pass
            elif dataset=="caltech101":
                pass
            elif dataset=="food101":
                dataset_dict["food101"] = {}
                for class_index, class_path in enumerate(os.listdir(os.path.join(cfg.dataset_path, "food101", "images"))):
                    dataset_dict["food101"][class_index] = []
                    for image_path in os.listdir(class_path):
                        dataset_dict["food101"][class_index].append(os.path.join(class_path, image_path))
            else:
                raise NotImplementedError

        return dataset_dict

        
        

class EpisodicSampler():
    def __init__(self,paths, n_shot = 1, n_query = 16) -> None:
        self.n_shot = n_shot
        self.n_query = n_query
        
    def __call__():
        pass
        
        
        
def main():
    folder_explorer = FolderExplorer(cfg.dataset_path)

    paths = folder_explorer()

    for dataset_name, list_classes in paths.items():
        print(dataset_name)

        print(list_classes[:3])

        print(list_classes[0][:3])

        print()

