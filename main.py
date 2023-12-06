import torch
from dataset import DatasetBuilder
from model import get_model
from config import cfg  # cfg.paths is a list of paths to the datasets
from classif.ncm import NCM
from tools import ResizeModulo
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from models.maskBlocks import Identity, Lost, DeepSpectralMethods, SAM, combine_masks

from augment.augmentations import crop_mask
from tools import PadAndResize

def baseline(cfg):
    sampler = DatasetBuilder(cfg)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(size="s",use_v2=False)
    model.to(device)
    model.eval()

    resize = T.Resize((224,224))

    transforms = T.Compose([
            PadAndResize(224),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225]) # imagenet mean and std
        ])
    
    identity = Identity()
    L_acc = []
    ncm = NCM()

    pbar = tqdm(range(cfg.n_runs), desc="Runs")

    for episode_idx in pbar:
        
        # new sample for each run
        episode = sampler() #episode is (dataset, classe, support/query, image_path)

        imagenet_sample = episode["imagenet"]

        support_images, temp_support_labels, query_images, temp_query_labels = imagenet_sample

        support_images = [resize(image) for image in support_images]    

        query_images = [resize(image) for image in query_images]


        # strategy: take the identity mask + the top k (config) sam masks 

        support_augmented_imgs = []
        support_labels = []

        for i, img in enumerate(support_images):
            masks_id = identity(img)
        
            masks = masks_id 
            support_augmented_imgs += [crop_mask(img, mask["segmentation"], z=0) for mask in masks]
            labels = [(temp_support_labels[i], i) for j in range(len(masks))]
            support_labels += labels
        
        query_augmented_imgs = []
        query_labels = []
        for i, img in enumerate(query_images):
            masks_id = identity(img)
            masks = masks_id
            query_augmented_imgs += [crop_mask(img, mask["segmentation"], z=0) for mask in masks]
            labels = [(temp_query_labels[i], i) for j in range(len(masks))]
            query_labels += labels
        
        support_augmented_imgs = [transforms(img).to(device) for img in support_augmented_imgs]
        query_augmented_imgs = [transforms(img).to(device) for img in query_augmented_imgs]

        support_tensor = torch.zeros((len(support_augmented_imgs), 384)) # size of the feature vector
        query_tensor = torch.zeros((len(query_augmented_imgs), 384))

        bs = cfg.batch_size
        
        with torch.inference_mode():
            for i in range(0, len(support_augmented_imgs), bs):
                inputs = torch.stack(support_augmented_imgs[i:i+bs])
                outputs = model(inputs)
                support_tensor[i:i+bs] = outputs

            for i in range(0, len(query_augmented_imgs), bs):
                inputs = torch.stack(query_augmented_imgs[i:i+bs])
                outputs = model(inputs)
                query_tensor[i:i+bs] = outputs

        acc = ncm(support_tensor, query_tensor, support_labels, query_labels)

        L_acc.append(acc)
        pbar.set_description(f"Last: {round(acc,2)}, avg: {round(np.mean(L_acc),2)}")

    print("Average accuracy: ", round(np.mean(L_acc),2), "std: ", round(np.std(L_acc),2))   
    print("All accuracies: ", np.round(L_acc,2))


def baseline_no_sam(cfg):
    sampler = DatasetBuilder(cfg)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(size="s",use_v2=False)
    model.to(device)
    model.eval()

    resize = T.Resize((224*2,224*2))

    transforms = T.Compose([
            ResizeModulo(patch_size=16, target_size=224),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225]) # imagenet mean and std
        ])
    
    identity = Identity()
    spectral = DeepSpectralMethods(model=model,
                                      n_eigenvectors=cfg.top_k_masks,
                                      lambda_color=10)

    L_acc = []
    ncm = NCM()

    pbar = tqdm(range(cfg.n_runs), desc="Runs")

    for episode_idx in pbar:
        
        # new sample for each run
        episode = sampler() #episode is (dataset, classe, support/query, image_path)

        imagenet_sample = episode["imagenet"]

        support_images, temp_support_labels, query_images, temp_query_labels = imagenet_sample

        support_images = [resize(image) for image in support_images]    

        query_images = [resize(image) for image in query_images]


        # strategy: take the identity mask + the top k (config) sam masks 

        support_augmented_imgs = []
        support_labels = []

        for i, img in enumerate(support_images):
            masks_id = identity(img)
            masks_spectral = spectral(img)
            masks = masks_id + masks_spectral[:cfg.top_k_masks]
            support_augmented_imgs += [crop_mask(img, mask["segmentation"], z=0) for mask in masks]
            labels = [(temp_support_labels[i], i) for j in range(len(masks))]
            support_labels += labels
        
        query_augmented_imgs = []
        query_labels = []
        for i, img in enumerate(query_images):
            masks_id = identity(img)
            masks_spectral = spectral(img)
            masks = masks_id + masks_spectral[:cfg.top_k_masks]
            query_augmented_imgs += [crop_mask(img, mask["segmentation"], z=0) for mask in masks]
            labels = [(temp_query_labels[i], i) for j in range(len(masks))]
            query_labels += labels
        
        support_augmented_imgs = [transforms(img).to(device) for img in support_augmented_imgs]
        query_augmented_imgs = [transforms(img).to(device) for img in query_augmented_imgs]

        support_tensor = torch.zeros((len(support_augmented_imgs), 384)) # size of the feature vector
        query_tensor = torch.zeros((len(query_augmented_imgs), 384))

        #bs = cfg.batch_size
        
        with torch.inference_mode():
            for i in range(len(support_augmented_imgs)):
                inputs = support_augmented_imgs[i].unsqueeze(0)
                outputs = model(inputs).squeeze(0)
                support_tensor[i] = outputs

            for i in range(len(query_augmented_imgs)):
                inputs = query_augmented_imgs[i].unsqueeze(0)
                outputs = model(inputs).squeeze(0)
                query_tensor[i] = outputs

        acc = ncm(support_tensor, query_tensor, support_labels, query_labels)

        L_acc.append(acc)
        pbar.set_description(f"Last: {round(acc,2)}, avg: {round(np.mean(L_acc),2)}")

    print("Average accuracy: ", round(np.mean(L_acc),2), "std: ", round(np.std(L_acc),2))   
    print("All accuracies: ", np.round(L_acc,2))
    

def main(cfg):
    sampler = DatasetBuilder(cfg)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(size="s",use_v2=False)
    model.to(device)
    model.eval()

    resize = T.Resize((224,224))

    transforms = T.Compose([
            ResizeModulo(patch_size=16, target_size=224),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225]) # imagenet mean and std
        ])
    
    identity = Identity()
    lost_deg_seed = Lost(alpha=0.95, k=100, model=model)
    lost_atn_seed = Lost(alpha=0.05, k=100, model=model)
    spectral = DeepSpectralMethods(model=model, 
                                   n_eigenvectors=15, 
                                   lambda_color=5)
    sam = SAM()

    L_acc = []
    ncm = NCM()

    pbar = tqdm(range(cfg.n_runs), desc="Runs")

    for episode_idx in pbar:
        
        # new sample for each run
        episode = sampler() #episode is (dataset, classe, support/query, image_path)

        imagenet_sample = episode["imagenet"]

        support_images, temp_support_labels, query_images, temp_query_labels = imagenet_sample

        support_images = [resize(image) for image in support_images]    

        query_images = [resize(image) for image in query_images]

        support_augmented_imgs = []
        support_labels = []

        for i, img in enumerate(support_images):
            masks_id = identity(img)
            mask_lost = lost_deg_seed(img) + lost_atn_seed(img) # concatenate the output of the two lost blocks
            approx_area = np.mean([mask["area"] for mask in mask_lost]) # average area of the lost masks

            # filter out masks that are too small
            mask_sam = sam(img)
            mask_sam_f = [mask for mask in mask_sam if mask["area"] > 0.4*approx_area] 
            # keep the same magnitude of area as the lost masks
            if len(mask_sam_f) > 2:
                mask_sam = mask_sam_f

            mask_spectral = spectral(img)
            
            masks_sam, masks_spectral, _ = combine_masks(mask_sam, mask_spectral, mask_lost, norm=True, postprocess=True)
            masks = masks_id + masks_sam[:cfg.top_k_masks] # + masks_spectral[:cfg.top_k_masks]
            support_augmented_imgs += [crop_mask(img, mask["segmentation"], z=0) for mask in masks]

            labels = [(temp_support_labels[i], i) for j in range(len(masks))] 
            support_labels += labels

        query_augmented_imgs = []
        query_labels = []

        for i, img in enumerate(query_images):
            masks_id = identity(img)
            mask_lost = lost_deg_seed(img) + lost_atn_seed(img)
            approx_area = np.mean([mask["area"] for mask in mask_lost])

            mask_sam = sam(img)
            mask_sam_f = [mask for mask in mask_sam if mask["area"] > 0.4*approx_area]
            if len(mask_sam_f) > 2:
                mask_sam = mask_sam_f
            
            mask_spectral = spectral(img)

            masks_sam, masks_spectral, _ = combine_masks(mask_sam, mask_spectral, mask_lost, norm=True, postprocess=True)
            masks = masks_sam[:cfg.top_k_masks] + masks_spectral[:cfg.top_k_masks]
            query_augmented_imgs += [crop_mask(img, mask["segmentation"], z=0) for mask in masks]

            labels = [(temp_query_labels[i], i) for j in range(len(masks))]
            query_labels += labels

        support_augmented_imgs = [transforms(img).to(device) for img in support_augmented_imgs]
        query_augmented_imgs = [transforms(img).to(device) for img in query_augmented_imgs]

        # size of the feature vector, WARNING: this is hardcoded for the model used
        support_tensor = torch.zeros((len(support_augmented_imgs), 384)) 
        query_tensor = torch.zeros((len(query_augmented_imgs), 384))

        #bs = cfg.batch_size
        
        with torch.inference_mode():
            for i in range(len(support_augmented_imgs)):
                inputs = support_augmented_imgs[i].unsqueeze(0)
                outputs = model(inputs).squeeze(0)
                support_tensor[i] = outputs

            for i in range(len(query_augmented_imgs)):
                inputs = query_augmented_imgs[i].unsqueeze(0)
                outputs = model(inputs).squeeze(0)
                query_tensor[i] = outputs

        acc = ncm(support_tensor, query_tensor, support_labels, query_labels)

        L_acc.append(acc)
        pbar.set_description(f"Last: {round(acc,2)}, avg: {round(np.mean(L_acc),2)}")

    print("Average accuracy: ", round(np.mean(L_acc),2), "std: ", round(np.std(L_acc),2))   
    print("All accuracies: ", np.round(L_acc,2))

                
if __name__ == "__main__":
    
    print("Config:", cfg.sampler)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", "-t", type=str, default="baseline", help="baseline, nosam, main")
    
    args = parser.parse_args()
    
    if args.type == "baseline":
        baseline(cfg)
    elif args.type == "nosam":
        baseline_no_sam(cfg)
    elif args.type == "main":
        main(cfg)