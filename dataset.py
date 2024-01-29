import os
import random as rd
from typing import Any
from config import cfg
import json
from PIL import Image

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
                            test_dataset_images_ids_class[class_name].append(os.path.join(cfg.paths["cub"], "images", line))

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
                    image_names = os.listdir(os.path.join(cfg.paths["cifarfs"], class_name))
                    dataset_dict["cifarfs"][class_name] = [os.path.join(cfg.paths["cifarfs"], class_name, image_name) for image_name in image_names]
            elif dataset=="fungi":
                dataset_dict["fungi"] = {}
                for class_name in os.listdir(os.path.join(cfg.paths["fungi"], "images")):
                    image_names = os.listdir(os.path.join(cfg.paths["fungi"], "images", class_name))
                    dataset_dict["fungi"][class_name] = [os.path.join(cfg.paths["fungi"], "images", class_name, image_name) for image_name in image_names]

            elif dataset=="caltech":
                dataset_dict["caltech"] = {}
                for class_name in os.listdir(os.path.join(cfg.paths["caltech"],"101_ObjectCategories")):
                    dataset_dict["caltech"][class_name] = os.listdir(os.path.join(cfg.paths["caltech"],"101_ObjectCategories", class_name))
                    dataset_dict["caltech"][class_name] = [img for img in dataset_dict["caltech"][class_name] if img.lower().endswith(".jpg") or img.lower().endswith(".png") or img.lower().endswith(".jpeg")]
                    dataset_dict["caltech"][class_name] = [os.path.join(cfg.paths["caltech"],"101_ObjectCategories", class_name, img) for img in dataset_dict["caltech"][class_name]]
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
        
    def __call__(self, seed_classes = None, seed_images = None) -> Any:
        """
        returns a dict where the key is the dataset name and the value is a dict of list

        seed_classes: int, seed for the random sampling of the classes
        seed_images: int, seed for the random sampling of the images

        E.g.: you want to sample from the same classes but different images, set seed_classes to a fixed value and seed_images to None
        """
        episode_dict = {}
        for dataset in self.paths:
            if seed_classes is not None:
                rd.seed(seed_classes)
            selected_classes = rd.sample(list(self.paths[dataset].keys()), self.n_ways[dataset])
            episode_dict[dataset] = {}
            for classe in selected_classes:
                episode_dict[dataset][classe] = {}
                if seed_images is not None:
                    rd.seed(seed_images)
                shuffle = rd.sample(self.paths[dataset][classe], min(self.n_shot+self.n_query, len(self.paths[dataset][classe])))
                episode_dict[dataset][classe]["support"] = shuffle[:self.n_shot]
                episode_dict[dataset][classe]["query"] = shuffle[self.n_shot:]
            
        return episode_dict
            
        
        
class DatasetBuilder():
    """
    return a dict where the key is the dataset name and the value is a tuple of 4 lists:
    support_images, support_labels, query_images, query_labels

    support_images: list of PIL images
    support_labels: list of labels
    query_images: list of PIL images
    query_labels: list of labels
    """
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        pass

    def __call__(self, seed_classes = None, seed_images = None) -> Any:
        folder_explorer = FolderExplorer(self.cfg.paths)
        paths = folder_explorer()
        sampler = EpisodicSampler(paths = paths,
                                n_query= self.cfg.sampler.n_queries,
                                n_ways = self.cfg.sampler.n_ways,
                                n_shot = self.cfg.sampler.n_shots)
        episode = sampler(seed_classes = seed_classes, seed_images = seed_images)
        # episode is (dataset, classe, support/query, image_path)
        dataset_dict = {}
        for dataset_name, list_classes in episode.items():
            support_images = [image_path for classe in list_classes for image_path in list_classes[classe]["support"]]
            support_labels = [classe for classe in list_classes for image_path in list_classes[classe]["support"]]
            query_images = [image_path for classe in list_classes for image_path in list_classes[classe]["query"]]
            query_labels = [classe for classe in list_classes for image_path in list_classes[classe]["query"]]
            dataset_dict[dataset_name] = (support_images, support_labels, query_images, query_labels)
        return dataset_dict
                
class COCOSampler():
    def __init__(self,cfg):
        self.path = cfg.paths["coco"]
        self.n_ways = cfg.sampler.n_ways["coco"]
        self.n_shots = cfg.sampler.n_shots
        self.n_queries = cfg.sampler.n_queries
    def __call__(self, seed_classes = None, seed_images = None):
        with open(f"{self.path}/annotations/instances_val2017.json", "r") as f:
            data = json.load(f)
        all_images = [(data["images"][i]["id"], data["images"][i]["file_name"]) for i in range(len(data["images"]))]
        all_annotations = [(data["annotations"][i]["category_id"],
                            data["annotations"][i]["image_id"],
                            data["annotations"][i]["bbox"])
                            for i in range(len(data["annotations"]))]
        categories = {}
        # count the number of images per category
        # problem here: one image can be in multiple categories so redundancy
        for i in range(len(all_annotations)):
            if all_annotations[i][0] not in categories:
                categories[all_annotations[i][0]] = set()
            categories[all_annotations[i][0]].add(all_annotations[i][1])
        valid_categories = [] # we want only categories with sufficient images to sample
        for category in categories:
            if len(categories[category]) >= 4*(self.n_shots+self.n_queries): # 4*(k+m) to not always sample the same images
                valid_categories.append(category)
        selected_categories = rd.sample(valid_categories, self.n_ways)
        selected_images = {}
        seen_images = set()
        for category in selected_categories:
            selected_images[category] = []
            # take k+m images randomly from categories[category] but we need to make sure that
            # the images are not in seen_imgs to avoid duplicates among categories
            category_images = list(categories[category])
            rd.shuffle(category_images)
            for img in category_images:
                if img not in seen_images:
                    selected_images[category].append(img)
                    seen_images.add(img)
                if len(selected_images[category]) == self.n_shots+self.n_queries:
                    break
            assert len(selected_images[category]) == self.n_shots+self.n_queries
        selected_annotations = {}
        for annotation in all_annotations:
            category_id, image_id, bbox = annotation
            if image_id in seen_images:
                if image_id not in selected_annotations:
                    selected_annotations[image_id] = []
                if category_id in selected_categories:
                    selected_annotations[image_id].append((category_id, bbox))
        for category in selected_categories:
            for img in selected_images[category]:
                selected_annotations[img] = (category, selected_annotations[img]) # (img_category, [(category, bbox), ...])
        for img in all_images:
            img_id, img_name = img
            if img_id in selected_annotations:
                img_path = f"{self.path}/images/val2017/{img_name}"
                selected_annotations[img_id] = (img_path, selected_annotations[img_id][0], selected_annotations[img_id][1])
        #selected_annotations = {img_id: (img_path, img_category, [(category, bbox), ...])}
        #group by category and split into support and query
        dataset = {}
        for img_id in selected_annotations:
            img_path, img_category, annotations = selected_annotations[img_id]
            if img_category not in dataset:
                dataset[img_category] = {}
                dataset[img_category]["support"] = []
                dataset[img_category]["query"] = []
            if len(dataset[img_category]["support"]) < self.n_shots:
                dataset[img_category]["support"].append(selected_annotations[img_id])
            else:
                dataset[img_category]["query"].append(selected_annotations[img_id])
        #dataset = {img_category: {"support": [(img_path, img_category, [(category, bbox), ...]), ...], "query": [img_path, ...]}, ...}
        #now we need to sample the query images
        return dataset

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

    imagenet_sample = episode["imagenet"]

    print(len(imagenet_sample))



def main_ter():
    dataset_builder = DatasetBuilder()

    new_dataset = dataset_builder()

    for dataset_name, list_classes in new_dataset.items():
        print(dataset_name)
        support_images, support_labels, query_images, query_labels = list_classes
        print("support images:  ", len(support_images))
        print("support labels:  ", len(support_labels))
        print("query images:  ", len(query_images))
        print("query labels:  ", len(query_labels))

def main_coco():
    coco_sampler = COCOSampler(cfg)
    coco_sample = coco_sampler()
    test_category = list(coco_sample.keys())[0]
    print(test_category)
    print("support")
    print(coco_sample[test_category]["support"])
    print("query")
    print(coco_sample[test_category]["query"])
if __name__== "__main__":
    main_coco()