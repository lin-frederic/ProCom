"""
These blocks extract the masks from the image and return it.

- Identity : the mask is the whole image
- Lost : the masks are the seed expansion set + the whole image
- SAM : the masks are the masks from SAM Automatic Mask Generator
- DeepSpectralMethod : the masks are the thresholded eigenvectors of the Laplacian matrix
- Combined : the masks are the masks from SAM AMG + the masks from DeepSpectralMethod, optimized to be similar to the masks from Lost


Args: PIL image of shape (H,W,3) on CPU, we take H = W = 224
Out : list of {mask, area}
"""

from torch import nn
import torch
import numpy as np
from model import get_model
from torchvision import transforms as T
import cv2
from PIL import Image

from models.lost import Lost as Lost_module
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from models.deepSpectralMethods import DSM

from tools import iou, focal_loss, dice_loss
from typing import Union

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, img):
        w, h = img.size
        mask = np.ones((w,h),dtype=np.uint8)
        mask[0,0] = 0 # remove one pixel to have a clean map
        return [{"segmentation": mask, "area": w*h}] # whole image

class Lost(nn.Module):
    def __init__(self,alpha, k, model = None):
        super().__init__()
        self.res = 224
        self.up = 2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if model is None:
            model = get_model(size="s",use_v2=False) # loads a DINOv1 model, size s
            model.to(self.device)
        self.lost = Lost_module(model=model,alpha=alpha,k=k*self.up**2)

    def clean(self, out, w, h):
        out = cv2.resize(out.astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST)
        # threshold the output to get a binary mask
        _, out = cv2.threshold(out, 128, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # get the largest connected component amongst the non-zero pixels

        components = cv2.connectedComponentsWithStats(out, connectivity=4)
        num_labels, labels, stats, centroids = components
        # identify the background label
        background_indexes = np.where(out == 0)
        background_label = np.median(labels[background_indexes])

        # sort the labels by area
        sorted_labels = np.argsort(stats[:, cv2.CC_STAT_AREA])[::-1] 
        # get the largest connected component
        largest_component_label = sorted_labels[0] if sorted_labels[0] != background_label else sorted_labels[1]

        # get the mask of the largest connected component
        mask = np.where(labels == largest_component_label, 1, 0).astype(np.uint8)

        return mask

    
    def forward(self, img):
        
        w, h = img.size

        new_w, new_h = int(w*self.up), int(h*self.up)

        img = T.Resize((new_h//16*16,new_w//16*16), antialias=True)(img)
        img = T.ToTensor()(img)
        img = img.unsqueeze(0).to(self.device)

        out = self.lost(img)
        mask = self.clean(out, w, h)

        return [{"segmentation": mask, "area": np.sum(mask)},]

class SAM(nn.Module):
    def __init__(self, size="b"):
        super().__init__()
        if size == "s":
            size = "b" # b is the smallest size
            print("Warning : SAM size s does not exist, using SAM size b instead")
        sizes = {
            "b" : "sam_vit_b_01ec64.pth",
            "l" : "sam_vit_l_0b3195.pth",
            "h" : "sam_vit_h_4b8939.pth",
        }

        sam = sam_model_registry[f"vit_{size}"](checkpoint=f"/nasbrain/f21lin/{sizes[size]}")
        print("SAM loaded")
        sam.to("cuda")

        self.AMG = SamAutomaticMaskGenerator(sam, 
                                             points_per_side=16,
                                             stability_score_thresh=0.82)
        
    def forward(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        masks = self.AMG.generate(img)
        masks = [{"segmentation": mask["segmentation"], 
                 "area": mask["area"]} for mask in masks]
        masks = sorted(masks, key=lambda x: x["area"], reverse=True) # sort by area (largest first)
        return masks

class DeepSpectralMethods(nn.Module):
    def __init__(self, model = None, n_eigenvectors=5, lambda_color=10):
        super().__init__()
        self. transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
        ])
        if model is None:
            model = get_model(size="s",use_v2=False)
        self.dsm = DSM(model=model, n_eigenvectors=n_eigenvectors, lambda_color=lambda_color)
    def clean(self, out, w, h):
        out = cv2.resize(out.astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST)
        # threshold the output to get a binary mask
        _, out = cv2.threshold(out, 128, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # get the largest connected component amongst the non-zero pixels

        components = cv2.connectedComponentsWithStats(out, connectivity=4)
        num_labels, labels, stats, centroids = components
        # identify the background label
        background_indexes = np.where(out == 0)
        background_label = np.median(labels[background_indexes])

        # sort the labels by area
        sorted_labels = np.argsort(stats[:, cv2.CC_STAT_AREA])[::-1] 
        # get the largest connected component
        largest_component_label = sorted_labels[0] if sorted_labels[0] != background_label else sorted_labels[1]

        # get the mask of the largest connected component
        mask = np.where(labels == largest_component_label, 1, 0).astype(np.uint8)

        return mask
    def forward(self, img):
        
        img = self.transforms(img).unsqueeze(0).to("cuda")
        eigenvectors = self.dsm(img) # returns a list of eigenvectors (arrays))

        masks = []

        for i in range(len(eigenvectors)):
            mask = eigenvectors[i]
            mask = cv2.resize(mask, (224,224), interpolation=cv2.INTER_NEAREST)
            _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            mask = np.where(mask > mask.mean(), 1, 0).astype(np.uint8)

            masks.append({"segmentation": mask, "area": np.sum(mask)})

        masks = sorted(masks, key=lambda x: x["area"], reverse=True) # sort by area
        for mask in masks:
            mask["segmentation"] = self.clean(mask["segmentation"], 224, 224)
        return masks
    
def combine_masks(masks_sam, masks_spectral, masks_lost, norm : Union[float, bool], postprocess=True):
    """
    masks_sam : list of masks from SAM
    masks_spectral : list of masks from DeepSpectralMethods
    masks_lost : list of masks from Lost
    loss : loss function
    norm : if True, normalize the losses (focal and dice) across the masks (before summing them)
    if float, then it is the coefficient of the focal loss i.e.
    loss = lambda x,y : dice_loss(x,y) + norm * focal_loss(x,y)
    postprocess : postprocess to have 
    
    returns : list of masks from SAM + DeepSpectralMethods +  lost, optimized to be similar to the masks from Lost (minimize the loss)

    --> intuitively, the best masks are the ones that are the most similar to the lost masks
    """
    def postprocess_area(masks, threshold=0.5):
    # invert mask to have an area less than 0.5 (or threshold) of the image
    # in place operation
        for mask in masks:
            if mask["area"] > threshold * mask["segmentation"].shape[0] * mask["segmentation"].shape[1]:
                mask["segmentation"] = 1 - mask["segmentation"]
                mask["area"] = np.sum(mask["segmentation"])

    def postprocess_loss(mask, reference, loss):
        # invert the mask if it minimizes the loss
        # in place operation
        mask = mask > 0
        reference = reference > 0
        inv = 1 - mask
        if loss(inv, reference) < loss(mask, reference):
            mask = inv
        return mask

    
    # postprocess the masks
    if postprocess is True:

        postprocess_area(masks_lost, threshold=0.5)

        reference = np.prod([mask["segmentation"] for mask in masks_lost], axis=0) # intersection of the lost masks
        reference = reference > 0

        """for mask in masks_sam:
            mask["segmentation"] = postprocess_loss(mask["segmentation"], reference, dice_loss)
        for mask in masks_spectral:
            mask["segmentation"] = postprocess_loss(mask["segmentation"], reference, dice_loss)"""

        thr = 0.7
        postprocess_area(masks_sam, threshold=thr)
        postprocess_area(masks_spectral, threshold=thr)

    # compute the losses
    # 2 losses : focal and dice
    losses_sam = np.zeros((len(masks_sam), len(masks_lost), 2)) 
    losses_spectral = np.zeros((len(masks_spectral), len(masks_lost), 2))

    for i in range(len(masks_lost)):
        for j in range(len(masks_sam)):
            losses_sam[j,i,0] = focal_loss(masks_sam[j]["segmentation"], masks_lost[i]["segmentation"])
            losses_sam[j,i,1] = dice_loss(masks_sam[j]["segmentation"], masks_lost[i]["segmentation"])
        for j in range(len(masks_spectral)):
            losses_spectral[j,i,0] = focal_loss(masks_spectral[j]["segmentation"], masks_lost[i]["segmentation"])
            losses_spectral[j,i,1] = dice_loss(masks_spectral[j]["segmentation"], masks_lost[i]["segmentation"])

    if norm is True:
        losses_sam[::,0] /= np.sum(losses_sam[:,:,0]) # focal / sam 
        losses_sam[::,1] /= np.sum(losses_sam[:,:,1]) # dice / sam

        losses_spectral[::,0] /= np.sum(losses_spectral[:,:,0]) # focal / spectral
        losses_spectral[::,1] /= np.sum(losses_spectral[:,:,1]) # dice / spectral
    
    else :
        losses_sam[::,0] *= norm # focal / sam
        losses_spectral[::,0] *= norm # focal / spectral
    
    losses_sam = np.sum(losses_sam, axis=2) # sum the losses, shape (len(masks_sam), len(masks_lost))
    losses_spectral = np.sum(losses_spectral, axis=2) # sum the losses, shape (len(masks_spectral), len(masks_lost))

    losses_spectral = np.sum(losses_spectral, axis=1) # sum the losses, shape (len(masks_spectral),)
    losses_sam = np.sum(losses_sam, axis=1) # sum the losses, shape (len(masks_sam),)

    # sort the masks by loss (argsort returns the indices)
    masks_sam = [masks_sam[i] for i in np.argsort(losses_sam)]
    masks_spectral = [masks_spectral[i] for i in np.argsort(losses_spectral)]

    return masks_sam, masks_spectral, masks_lost 

import os
import matplotlib.pyplot as plt
import uuid
import argparse
from tqdm import tqdm
import time
def main(n, seed):
    start = time.time()
    if seed > 0:
        np.random.seed(seed)

    # instantiate the blocks ##############################################
    identity = Identity()
    sam = SAM(size="s")

    dino_model = get_model(size="s",use_v2=False).to("cuda") # shared model for lost and dsm

    lost_deg_seed = Lost(alpha=0.9, k=100, model=dino_model)
    lost_atn_seed = Lost(alpha=0.1, k=100, model=dino_model)
    
    spectral = DeepSpectralMethods(model=dino_model, n_eigenvectors=15, lambda_color=5)

    ######################################################################

    setup_time = time.time()

    for ind in tqdm(range(n)):
        
        root = "/nasbrain/datasets/imagenet/images/val/"
        folder = np.random.choice(os.listdir(root))
        path = os.path.join(root,folder)

        img = Image.open(os.path.join(path,np.random.choice(os.listdir(path)))).convert("RGB")
        img = T.Resize((224,224), antialias=True)(img)

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
        
        masks_sam, masks_spectral, masks_lost = combine_masks(mask_sam, mask_spectral, mask_lost, norm=True, postprocess=True)

            
        masks = [masks_id, masks_lost, masks_sam, masks_spectral]
        titles = ["Id", "Lost", "SAM", "DSM"]

        n_ = 5
        fig, axes = plt.subplots(len(masks)+1, n_, figsize=(10,10))
        axes[0,0].imshow(img)
        axes[0,0].set_title("Original image")
        axes[0,0].axis("off")

        for i, (mask, title) in enumerate(zip(masks, titles)):
            for j in range(min(n_,len(mask))):
                axes[i+1,j].imshow(mask[j]["segmentation"])
                if "loss" in mask[j].keys():
                    axes[i+1,j].set_title(f"{title}'s loss: {mask[j]['loss']:.1e}")
                else:
                    axes[i+1,j].set_title(f"{title}'s area: {mask[j]['area']:.1e}")
                
        for i in range(len(masks)+1):
            for j in range(n_):
                axes[i,j].axis("off")


        plt.tight_layout()

        name = str(uuid.uuid4())
        plt.savefig(f"temp/img_{name}.png")
        plt.close()

    end = time.time()

    print("done")
    print(f"setup time : {setup_time-start:.4g} s")
    print(f"total time : {end-start:.4g} s")
    print(f"time / img : {(end-setup_time)/n:.4g} s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n", type=int, default=5)
    parser.add_argument("-s", "--seed", type=int, default=-1)
    n = parser.parse_args().n
    seed = parser.parse_args().seed
    main(n, seed)



