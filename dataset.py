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
                train_test_file_path = os.path.join(dataset, "train_test_split.txt")
                image_file_path = os.path.join(dataset, "images.txt")
                class_file_path = os.path.join(dataset, "image_class_labels.txt")

                with open(train_test_file_path, "r") as train_test_file, open(image_file_path, "r") as image_file, open(class_file_path, "r") as class_file:
                    for train_test_line, image_line, class_line in izip(train_test_file, image_file, class_file):
                        train_test_line = train_test_line.strip().split(" ")
                        image_line = image_line.strip().split(" ")
                        class_line = class_line.strip().split(" ")
                        if train_test_line[1] == "1":
                            dataset_dict["cub"][class_line[1]] = os.path.join(cfg.dataset_path, "images", image_line[1])

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

