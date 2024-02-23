import torch
from dataset import DatasetBuilder, COCOSampler, PascalVOCSampler, ImageNetLocSampler, CUBSampler
from model import get_model
from config import cfg  # cfg.paths is a list of paths to the datasets
from classif.matching import MatchingClassifier, NCM
from classif.linear import MyLinear
from tools import ResizeModulo
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from models.maskBlocks import Identity, DeepSpectralMethods, SAM, combine_masks, filter_masks
from uuid import uuid4
from augment.augmentations import crop_mask, crop
from tools import PadAndResize
from PIL import Image
import os
import json
import wandb

import matplotlib.pyplot as plt

from models.DSM_SAM import DSM_SAM
from model import get_sam_model, CachedSamPredictor
from segment_anything import SamPredictor
from models.deepSpectralMethods import DSM



def main_coco(cfg):
    coco_sampler = COCOSampler(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(size="s",use_v2=False).to(device)

    transforms = T.Compose([
            ResizeModulo(patch_size=16, target_size=224, tensor_out=True),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225]) # imagenet mean and std
        ])
    
    # coco dataset

    L_acc = []
    ncm = MatchingClassifier(seed=42)

    pbar = tqdm(range(cfg.n_runs), desc="Runs")

    for episode_idx in pbar:
        dataset = coco_sampler(seed_classes=episode_idx, seed_images=episode_idx)

        support_images, temp_support_labels, query_images, temp_query_labels, annotations = coco_sampler.format(dataset)
        
        filtered_annotations = coco_sampler.filter_annotations(annotations, filter=True) # (quality annotations)
        unfiltered_annotations = coco_sampler.filter_annotations(annotations, filter=False) # (noisy annotations)

        support_augmented_imgs = []
        support_labels = []

        for i, img_path in enumerate(support_images):
            img = Image.open(img_path).convert("RGB")
            bboxes = unfiltered_annotations[img_path] # list of bboxes
            support_augmented_imgs += [img] # add the original image
            for bbox in bboxes:
                #bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]] # convert to [x1,y1,x2,y2]
                support_augmented_imgs += [crop(img,bbox)]

            labels = [(temp_support_labels[i], i) for j in range(len(bboxes)+1)] #bounding box + original image
            support_labels += labels
        """# plot the support augmented images
        for img in support_augmented_imgs:
            img.save(f"results/{uuid4()}.png")
            
        exit()"""
            
        
        
        query_augmented_imgs = []
        query_labels = []

        for i, img_path in enumerate(query_images):
            img = Image.open(img_path).convert("RGB")
            """bboxes = unfiltered_annotations[img_path] # list of bboxes
            
            for bbox in bboxes:
                #bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
                query_augmented_imgs += [crop(img, bbox)]
"""
            query_augmented_imgs += [img]
            labels = [(temp_query_labels[i], i) for j in range(1)]
            #range(len(bboxes))]
            query_labels += labels

        support_augmented_imgs = [transforms(img).to(device) for img in support_augmented_imgs]
        query_augmented_imgs = [transforms(img).to(device) for img in query_augmented_imgs]

        support_tensor = torch.zeros((len(support_augmented_imgs), 384)) # size of the feature vector WARNING: hardcoded
        query_tensor = torch.zeros((len(query_augmented_imgs), 384))

        with torch.inference_mode():
            for i in range(len(support_augmented_imgs)):
                inputs = support_augmented_imgs[i].unsqueeze(0)
                outputs = model(inputs).squeeze(0)
                support_tensor[i] = outputs

            for i in range(len(query_augmented_imgs)):
                inputs = query_augmented_imgs[i].unsqueeze(0)
                outputs = model(inputs).squeeze(0)
                query_tensor[i] = outputs
        
        acc = ncm(support_tensor, query_tensor, support_labels, query_labels, use_cosine=True)

        L_acc.append(acc)
        pbar.set_description(f"Last: {round(acc,2)}, avg: {round(np.mean(L_acc),2)}")

        if cfg.wandb:
            wandb.log({"running_accuracy": acc,
                        "average_accuracy": np.mean(L_acc),
                       })
            
    print("Average accuracy: ", round(np.mean(L_acc),2), "std: ", round(np.std(L_acc),2))
    print("All accuracies: ", np.round(L_acc,2))


    use_AMG = cfg.use_AMG
    imagenetloc_sampler = ImageNetLocSampler(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(size="s",use_v2=False).to(device)
    if use_AMG:
        print("Using AMG")
        amg = SAM("b")
    
    else:
        print("Using hierarchical")
        dsm_model = DSM(model=model, # same model as the one used for the classification
                        n_eigenvectors=cfg.dsm.n_eigenvectors,
                        lambda_color=cfg.dsm.lambda_color)
        dsm_model.to(device)
        sam = get_sam_model(size="b").to(device)  
        sam_model = CachedSamPredictor(sam_model = sam, 
                                    path_to_cache=os.path.join(cfg.sam_cache, "embeddings", cfg.dataset),
                                    json_cache=os.path.join(cfg.sam_cache, "embeddings", cfg.dataset, "cache.json"))
        hierarchical = DSM_SAM(dsm_model, sam_model, 
                            nms_thr=cfg.hierarchical.nms_thr,
                            area_thr=cfg.hierarchical.area_thr,
                            target_size=224*2,)

    transforms = T.Compose([
            ResizeModulo(patch_size=16, target_size=224, tensor_out=True),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225]) # imagenet mean and std
        ])

    L_acc = []
    ncm = MatchingClassifier(seed=42)
    linear = MyLinear()

    pbar = tqdm(range(cfg.n_runs), desc="Runs")

    for episode_idx in pbar:
        support_images, temp_support_labels, query_images, temp_query_labels, annotations = imagenetloc_sampler(seed_classes=episode_idx, seed_images=episode_idx)
        filtered_annotations = imagenetloc_sampler.filter_annotations(annotations, filter=True) # (quality annotations)
        unfiltered_annotations = imagenetloc_sampler.filter_annotations(annotations, filter=False)
        support_augmented_imgs = []
        support_labels = []
        for i, img_path in enumerate(support_images):
            img = Image.open(img_path).convert("RGB")
            support_augmented_imgs += [img]
            bboxes = filtered_annotations[img_path] # list of bboxes
            for bbox in bboxes:
                # convert to [x,y,w,h]
                bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
                support_augmented_imgs += [crop(img,bbox,dezoom=cfg.dezoom)]
            
            labels = [(temp_support_labels[i], i) for j in range(len(bboxes)+1)] #bounding box + original image
            support_labels += labels

        query_augmented_imgs = []
        query_labels = []

        for i, img_path in enumerate(query_images):
            img = Image.open(img_path).convert("RGB")

            if use_AMG:
                resized_img = ResizeModulo(patch_size=16, target_size=224*2, tensor_out=False)(img) 
                # same size as the hierarchical method
                masks = amg.forward(img = resized_img)
                masks = [mask["segmentation"] for mask in masks if mask["area"] > mask["segmentation"].shape[0]*mask["segmentation"].shape[1]*0.05]
                # discard masks that are too small (less than 5% of the image)
            
            else:
                masks, _, resized_img = hierarchical.forward(img = img, 
                                                path_to_img=img_path,
                                                sample_per_map=cfg.hierarchical.sample_per_map,
                                                temperature=cfg.hierarchical.temperature)
                masks = masks.detach().cpu().numpy()

            query_augmented_imgs += [resized_img]
            query_augmented_imgs += [crop_mask(resized_img, mask, dezoom=cfg.dezoom) for mask in masks]

            labels = [(temp_query_labels[i], i) for j in range(len(masks)+1)] # bounding box + original image
            query_labels += labels

        support_augmented_imgs = [transforms(img).to(device) for img in support_augmented_imgs]
        query_augmented_imgs = [transforms(img).to(device) for img in query_augmented_imgs]
        augmentations = [T.RandomHorizontalFlip(p=1), T.RandomVerticalFlip(p=1), T.RandomRotation(90)]

        for i in range(len(support_augmented_imgs)):
            img = support_augmented_imgs[i]
            label = support_labels[i]
            for i in range(0):
                for aug in augmentations:
                    img = aug(img)
                    support_augmented_imgs.append(img)
                    support_labels.append(label)

        support_tensor = torch.zeros((len(support_augmented_imgs), 384)) # size of the feature vector WARNING: hardcoded
        query_tensor = torch.zeros((len(query_augmented_imgs), 384))

        with torch.inference_mode():
            for i in range(len(support_augmented_imgs)):
                inputs = support_augmented_imgs[i].unsqueeze(0)
                outputs = model(inputs).squeeze(0)
                support_tensor[i] = outputs

            for i in range(len(query_augmented_imgs)):
                inputs = query_augmented_imgs[i].unsqueeze(0)
                outputs = model(inputs).squeeze(0)
                query_tensor[i] = outputs

        use_linear = True
        if use_linear:
            acc = linear(support_tensor, query_tensor, support_labels, query_labels, temp_query_labels, encode_labels=True)
        else:
            acc = ncm(support_tensor, query_tensor, support_labels, query_labels, use_cosine=False)
        L_acc.append(acc)
        pbar.set_description(f"Last: {round(acc,2)}, avg: {round(np.mean(L_acc),2)}")
        if cfg.wandb:
            wandb.log({"running_accuracy": acc,
                        "average_accuracy": np.mean(L_acc),
                       })
    print("Average accuracy: ", round(np.mean(L_acc),2), "std: ", round(np.std(L_acc),2))
    print("All accuracies: ", np.round(L_acc,2))

def main_loc(cfg):
    dataset = cfg.dataset.lower() # imagenetloc, CUBloc, pascalVOC
    if dataset == "imagenetloc":
        sampler = ImageNetLocSampler(cfg)
    elif dataset == "cubloc":
        sampler = CUBSampler(cfg)
    elif dataset == "pascalvoc":
        sampler = PascalVOCSampler(cfg)
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(size="s",use_v2=False).to(device)
    
    assert cfg.setting.query   in ["whole", "AMG", "hierarchical", "filtered", "unfiltered"]
    assert cfg.setting.support in ["whole", "filtered", "unfiltered"]
    
    if cfg.setting.query == "AMG":
        print("Using AMG")
        amg = SAM("b")
        
    if cfg.setting.query == "hierarchical":
        print("Using hierarchical")
        dsm_model = DSM(model=model, # same model as the one used for the classification
                        n_eigenvectors=cfg.dsm.n_eigenvectors,
                        lambda_color=cfg.dsm.lambda_color)
        dsm_model.to(device)
        sam = get_sam_model(size="b").to(device)
        sam_model = CachedSamPredictor(sam_model = sam, 
                                    path_to_cache=os.path.join(cfg.sam_cache, "embeddings", cfg.dataset),
                                    json_cache=os.path.join(cfg.sam_cache, "embeddings", cfg.dataset, "cache.json"))
        hierarchical = DSM_SAM(dsm_model, sam_model, 
                            nms_thr=cfg.hierarchical.nms_thr,
                            area_thr=cfg.hierarchical.area_thr,
                            target_size=224*2,) 
    
    transforms = T.Compose([
            ResizeModulo(patch_size=16, target_size=224, tensor_out=True),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225]) # imagenet mean and std
        ])
    
    L_acc = []
    if cfg.classifier == "matching":
        classifier = MatchingClassifier(seed=42)
    elif cfg.classifier == "ncm":
        classifier = NCM(seed=42)
    
    pbar = tqdm(range(cfg.n_runs), desc="Runs")
    
    for episode_idx in pbar:
        
        support_images, temp_support_labels, query_images, temp_query_labels, annotations = sampler(seed_classes=episode_idx, seed_images=episode_idx)
        filtered_annotations = sampler.filter_annotations(annotations, filter=True)
        unfiltered_annotations = sampler.filter_annotations(annotations, filter=False)
        
        support_augmented_imgs = []
        support_labels = []
        
        for i, img_path in enumerate(support_images):
            img = Image.open(img_path).convert("RGB")
            
            if cfg.setting.support == "whole":
                support_augmented_imgs += [img]
                labels = [(temp_support_labels[i], i) for j in range(1)]
                
            elif cfg.setting.support in ["filtered", "unfiltered"]:
                bboxes = filtered_annotations[img_path] if cfg.setting.support == "filtered" else unfiltered_annotations[img_path]
                support_augmented_imgs += [img]
                for bbox in bboxes:
                    if dataset == "imagenetloc":
                        bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
                    elif dataset == "cubloc":
                        bbox = [bbox[0], bbox[1], bbox[2], bbox[3]] # no modification
                    elif dataset == "pascalvoc":
                        bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
                
                    support_augmented_imgs += [crop(img,bbox,dezoom=cfg.dezoom)]
                labels = [(temp_support_labels[i], i) for j in range(len(bboxes)+1)]
                
            support_labels += labels
        
        query_augmented_imgs = []
        query_labels = []
        
        for i, img_path in enumerate(query_images):
            img = Image.open(img_path).convert("RGB")
            
            if cfg.setting.query == "whole":
                query_augmented_imgs += [img]
                labels = [(temp_query_labels[i], i) for j in range(1)]
            
            elif cfg.setting.query in ["filtered", "unfiltered"]:
                bboxes = filtered_annotations[img_path] if cfg.setting.query == "filtered" else unfiltered_annotations[img_path]
                query_augmented_imgs += [img]
                for bbox in bboxes:
                    if dataset == "imagenetloc":
                        bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
                    elif dataset == "cubloc":
                        bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
                    elif dataset == "pascalvoc":
                        bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
                        
                    query_augmented_imgs += [crop(img, bbox, dezoom=cfg.dezoom)]
                labels = [(temp_query_labels[i], i) for j in range(len(bboxes)+1)]
            
            elif cfg.setting.query == "AMG":
                resized_img = ResizeModulo(patch_size=16, target_size=224*2, tensor_out=False)(img)
                masks = amg.forward(img = resized_img)
                masks = [mask["segmentation"] for mask in masks if mask["area"] > mask["segmentation"].shape[0]*mask["segmentation"].shape[1]*0.05]
                query_augmented_imgs += [resized_img]
                query_augmented_imgs += [crop_mask(resized_img, mask, dezoom=cfg.dezoom) for mask in masks]
                labels = [(temp_query_labels[i], i) for j in range(len(masks)+1)]
                
            elif cfg.setting.query == "hierarchical":
                masks, _, resized_img = hierarchical.forward(img = img, 
                                                path_to_img=img_path,
                                                sample_per_map=cfg.hierarchical.sample_per_map,
                                                temperature=cfg.hierarchical.temperature)
                masks = masks.detach().cpu().numpy()
                query_augmented_imgs += [resized_img]
                query_augmented_imgs += [crop_mask(resized_img, mask, dezoom=cfg.dezoom) for mask in masks]
                labels = [(temp_query_labels[i], i) for j in range(len(masks)+1)]
                
            query_labels += labels
            
        support_augmented_imgs = [transforms(img).to(device) for img in support_augmented_imgs]
        query_augmented_imgs = [transforms(img).to(device) for img in query_augmented_imgs]
        
        support_tensor = torch.zeros((len(support_augmented_imgs), 384)) # size of the feature vector WARNING: hardcoded
        query_tensor = torch.zeros((len(query_augmented_imgs), 384))
        
        with torch.inference_mode():
            for i in range(len(support_augmented_imgs)):
                inputs = support_augmented_imgs[i].unsqueeze(0)
                outputs = model(inputs).squeeze(0)
                support_tensor[i] = outputs

            for i in range(len(query_augmented_imgs)):
                inputs = query_augmented_imgs[i].unsqueeze(0)
                outputs = model(inputs).squeeze(0)
                query_tensor[i] = outputs
        acc = classifier(support_tensor, query_tensor, support_labels, query_labels, use_cosine=True)
        L_acc.append(acc)
        pbar.set_description(f"Last: {round(acc,2)}, avg: {round(np.mean(L_acc),2)}")
        if cfg.wandb:
            wandb.log({"running_accuracy": acc,
                        "average_accuracy": np.mean(L_acc),
                       })
            
    print("Average accuracy: ", round(np.mean(L_acc),2), "std: ", round(np.std(L_acc),2))
    print("All accuracies: ", np.round(L_acc,2))   

def main_seed(cfg, seed): # reproduce a run with a specific seed
    # this is used to visualize the matching between the masks in the ncm
    sampler = DatasetBuilder(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(size="s",use_v2=False)
    model.to(device)
    dsm_model = DSM(model=model, n_eigenvectors=5) # same model as the one used for the classification
    dsm_model.to(device)
    sam = get_sam_model(size="b").to(device)
    sam_model = SamPredictor(sam)
    hierarchical = DSM_SAM(dsm_model, sam_model, nms_thr=0.4)
    resize = ResizeModulo(patch_size=16, target_size=224, tensor_out=False)
    transforms = T.Compose([
            ResizeModulo(patch_size=16, target_size=224, tensor_out=True),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225]) # imagenet mean and std
        ])
    dataset = cfg.dataset
    ncm = MatchingClassifier(seed=42)
    episode = sampler(seed_classes=seed, seed_images=seed)
    sample = episode[dataset]
    support_images, temp_support_labels, query_images, temp_query_labels = sample
    support_augmented_imgs = []
    support_labels = []
    for i, img_path in enumerate(tqdm(support_images)):
        img = resize(Image.open(img_path).convert("RGB"))
        masks, _ = hierarchical(img, sample_per_map=10, temperature=255*0.1)
        masks = masks.detach().cpu().numpy()
        #add the identity mask
        masks = np.concatenate([np.ones((1,masks.shape[1],masks.shape[2])), masks], axis=0)
        #masks = masks[:cfg.top_k_masks]
        support_augmented_imgs += [crop_mask(img, mask, dezoom=0.1) for mask in masks]
        labels = [(temp_support_labels[i], i) for j in range(len(masks))]
        support_labels += labels
    query_augmented_imgs = []
    query_labels = []
    for i, img_path in enumerate(tqdm(query_images)):
        img = resize(Image.open(img_path).convert("RGB"))
        masks, _ = hierarchical(img, sample_per_map=10, temperature=255*0.1)
        masks = masks.detach().cpu().numpy()
        #add the identity mask
        masks = np.concatenate([np.ones((1,masks.shape[1],masks.shape[2])), masks], axis=0)
        #masks = masks[:cfg.top_k_masks]
        query_augmented_imgs += [crop_mask(img, mask, dezoom=0.1) for mask in masks]
        labels = [(temp_query_labels[i], i) for j in range(len(masks))]
        query_labels += labels
    support_augmented_imgs = [transforms(img).to(device) for img in support_augmented_imgs]
    query_augmented_imgs = [transforms(img).to(device) for img in query_augmented_imgs]
    support_tensor = torch.zeros((len(support_augmented_imgs), 384)) # size of the feature vector
    query_tensor = torch.zeros((len(query_augmented_imgs), 384))
    with torch.inference_mode():
        for i in tqdm(range(len(support_augmented_imgs))):
            inputs = support_augmented_imgs[i].unsqueeze(0)
            outputs = model(inputs).squeeze(0)
            support_tensor[i] = outputs
        for i in tqdm(range(len(query_augmented_imgs))):
            inputs = query_augmented_imgs[i].unsqueeze(0)
            outputs = model(inputs).squeeze(0)
            query_tensor[i] = outputs
    # convert augmented images to plottable images
    support_augmented_imgs = [img.squeeze(0).permute(1,2,0).cpu().numpy() for img in support_augmented_imgs]
    query_augmented_imgs = [img.squeeze(0).permute(1,2,0).cpu().numpy() for img in query_augmented_imgs]
    # support_augmented_imgs and query_augmented_imgs are lists of masked images
    # support images and query images are lists of original images
    to_display = [support_augmented_imgs, query_augmented_imgs, support_images, query_images]
    acc = ncm(support_tensor, query_tensor, support_labels, query_labels, use_cosine=True,to_display=to_display)
        
    print("Accuracy: ", round(acc,2))


                
if __name__ == "__main__":
    
    print("Config:", cfg.sampler)

    """
    python main.py -t baseline -d imagenet -w
    python main.py -t hierarchical -d imagenet -w
    python main.py -t main -d imagenet -w
    python main.py -t coco -d coco -w
    python main.py -t pascalVOC -d pascalVOC -w
    python main.py -t seed -d imagenet -w -s 0
    python main.py -t imagenetloc -d imagenetloc -w
    python main.py -t CUBloc -d CUBloc -w
    """
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", "-t", type=str, default="baseline", help="baseline, main, hierarchical")
    parser.add_argument("--wandb", "-w", action="store_true", help="use wandb")
    parser.add_argument("--dataset", "-d", type=str, default="imagenet", help="imagenet, cub, caltech, food, cifarfs, fungi, flowers, pets")
    parser.add_argument("--seed", "-s", type=int, default=0, help="seed for the run")
    parser.add_argument("--message", "-m", type=str, default="", help="message for the run, only used with wandb")
    args = parser.parse_args()

    if args.type == "coco":
        args.dataset = "coco"

    cfg["type"] = args.type
    cfg["dataset"] = args.dataset

    if args.wandb:
        wandb.login()
        wandb.init(project="procom-transformers", entity="procom", notes=args.message)
        cfg["wandb"] = True
        wandb.config.update(cfg)
    
    

    
    if args.type == "seed":
        main_seed(cfg, args.seed)
        
    elif args.type == "loc":
        main_loc(cfg)

    elif args.type == "coco":
        main_coco(cfg)


    else:
        raise ValueError(f"Unknown type of experiment: {args.type}")