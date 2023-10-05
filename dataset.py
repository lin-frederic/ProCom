import os
import random as rd
from typing import Any
from config import cfg
import json


class FolderExplorer():
    def __init__(self, dataset_paths) -> None:
        self.dataset_paths = dataset_paths
    
    def __call__(self) -> Any:
        # returns a dict where the key is the dataset name and the value is a dict of list
        
        # initialize the dict
        dataset_dict = {}

        for dataset in self.dataset_paths:
            if dataset=="imagenet":
                dataset_dict["imagenet"]={}
                for class_name in os.listdir(cfg.paths["imagenet"]):
                    dataset_dict["imagenet"][class_name] = [os.path.join(cfg.paths["imagenet"], class_name, image_name) for image_name in os.listdir(os.path.join(cfg.paths["imagenet"], class_name))]
                    

            elif dataset=="cub":
                with open(os.path.join(cfg.paths["cub"], "train_test_split.txt")) as f:
                    train_test_split = f.readlines()
                    # filter train dataset with the image ids
                    test_dataset_image_ids = [line.split()[0] for line in train_test_split if line.split()[1]=="1"]
                    

                    test_dataset_images_ids_class = {}
                    
                with open(os.path.join(cfg.paths["cub"], "images.txt")) as f:
                    images = f.readlines()
                    # filter the train dataset with the image ids and class ids
                    for line in images :
                        if line.split()[0] in test_dataset_image_ids:
                            line = line.split()[1]
                            class_name= line.split('/')[0]
                            if class_name not in test_dataset_images_ids_class:
                                test_dataset_images_ids_class[class_name] = []
                            test_dataset_images_ids_class[class_name].append(line)

                dataset_dict["cub"] = test_dataset_images_ids_class # {class: [image_id, ...]}

            elif dataset=="cifarfs":
                dataset_dict["cifarfs"] = {}
                # folder structure: path_to_cifarfs/(meta_train/meta_test)/class_name/image_name
                metas = ["meta-test", "meta-val"]
                class_names = []
                for meta in metas:
                    for class_name in os.listdir(os.path.join(cfg.paths["cifarfs"], meta)):
                        class_names.append(os.path.join(meta, class_name))
                for class_name in class_names:
                    dataset_dict["cifarfs"][class_name] = os.listdir(os.path.join(cfg.paths["cifarfs"], class_name))  
            elif dataset=="fungi":
                dataset_dict["fungi"] = {}
                for class_name in os.listdir(os.path.join(cfg.paths["fungi"], "images")):
                    dataset_dict["fungi"][class_name] = os.listdir(os.path.join(cfg.paths["fungi"], "images", class_name))
                
            elif dataset=="mscoco": #j'ai pas les droits
                pass
            elif dataset=="caltech":
                dataset_dict["caltech"] = {}
                for class_name in os.listdir(os.path.join(cfg.paths["caltech"],"101_ObjectCategories")):
                    dataset_dict["caltech"][class_name] = os.listdir(os.path.join(cfg.paths["caltech"],"101_ObjectCategories", class_name))
                    dataset_dict["caltech"][class_name] = [img for img in dataset_dict["caltech"][class_name] if img.lower().endswith(".jpg") or img.lower().endswith(".png") or img.lower().endswith(".jpeg")]
            elif dataset=="food":
                dataset_dict["food"] = {}
                with open(os.path.join(cfg.paths["food"], "split_zhou_Food101.json"), "r") as split_file:
                    split_dict = json.load(split_file)
                    for item in split_dict["val"]:
                        image_name = item[0]
                        class_index = item[1]
                        if class_index not in dataset_dict["food"]:
                            dataset_dict["food"][class_index] = []
                        dataset_dict["food"][class_index].append(os.path.join(cfg.paths["food"], "images", image_name))
            elif dataset=="flowers":
                dataset_dict["flowers"] = {}
                with open(os.path.join(cfg.paths["flowers"], "split_zhou_OxfordFlowers.json"), "r") as split_file:
                    split_dict = json.load(split_file)
                    for item in split_dict["val"]:
                        image_name = item[0]
                        class_index = item[1]
                        if class_index not in dataset_dict["flowers"]:
                            dataset_dict["flowers"][class_index] = []
                        dataset_dict["flowers"][class_index].append(os.path.join(cfg.paths["flowers"], "jpg", image_name))
            elif dataset=="pets":
                dataset_dict["pets"] = {}
                with open(os.path.join(cfg.paths["pets"], "split_zhou_OxfordPets.json"), "r") as split_file:
                    split_dict = json.load(split_file)
                    for item in split_dict["val"]:
                        image_name = item[0]
                        class_index = item[1]
                        if class_index not in dataset_dict["pets"]:
                            dataset_dict["pets"][class_index] = []
                        dataset_dict["pets"][class_index].append(os.path.join(cfg.paths["pets"], "images", image_name))

                
            else:
                print(f"Dataset {dataset} not found")

        return dataset_dict

        
        

class EpisodicSampler():
    #for each dataset, sample n_ways classes, then sample n_shot images for each class, then sample n_query images for each class
    def __init__(self,paths, n_shot = 1,n_ways = {dataset:5 for dataset in cfg.paths.keys()}, n_query = 16) -> None:
        self.n_shot = n_shot
        self.n_query = n_query
        self.paths = paths
        self.n_ways = n_ways
        
    def __call__(self):
        episode_dict = {}
        for dataset in self.paths:
            selected_classes = rd.sample(list(self.paths[dataset].keys()), self.n_ways[dataset])
            episode_dict[dataset] = {}
            for classe in selected_classes:
                episode_dict[dataset][classe] = {}
                shuffle = rd.sample(self.paths[dataset][classe], min(self.n_shot+self.n_query, len(self.paths[dataset][classe])))
                episode_dict[dataset][classe]["support"] = shuffle[:self.n_shot]
                episode_dict[dataset][classe]["query"] = shuffle[self.n_shot:]
            
        return episode_dict
        
        
        
    
def main():
    folder_explorer = FolderExplorer(cfg.paths)

    paths = folder_explorer()

    for dataset_name, list_classes in paths.items():
        print(dataset_name)
        print(len(list_classes))
        if len(list_classes) > 0:
            print(len(list_classes[list(list_classes.keys())[0]]))
            for classe in list_classes:
                if not (all(img.lower().endswith(".jpg") or img.lower().endswith(".png") or img.lower().endswith(".jpeg") for img in list_classes[classe])):
                    print("Not all images are jpg or png")
                    print(list_classes[classe])
                    break

        print()

def main_bis():
    folder_explorer = FolderExplorer(cfg.paths)

    paths = folder_explorer()

    episodic_sampler = EpisodicSampler(paths, 
                                       n_ways = {dataset:2 for dataset in cfg.paths.keys()},
                                        n_shot = 1,
                                        n_query = 2
                                       )

    episode = episodic_sampler()

    for dataset_name, list_classes in episode.items():
        print(dataset_name)
        print(list_classes)
        print()


if __name__== "__main__":
    main_bis()