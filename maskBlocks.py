"""
These blocks extract the masks from the image and return it.

- Identity : the mask is the whole image
- Lost : the masks are the seed expansion set + the whole image
- SAM : the masks are the masks from SAM Automatic Mask Generator
- DeepSpectralMethod : the masks are the thresholded eigenvectors of the Laplacian matrix


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

from lost import Lost as Lost_module
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from deepSpectralMethods import DSM

from tools import iou, focal_loss, dice_loss

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
    def __init__(self, model = None, n_eigenvectors=5):
        super().__init__()
        self. transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
        ])
        if model is None:
            model = get_model(size="s",use_v2=False)
        self.dsm = DSM(model=model, n_eigenvectors=n_eigenvectors)
    def forward(self, img):
        
        img = self.transforms(img).unsqueeze(0).to("cuda")
        eigenvectors = self.dsm(img) # returns a list of eigenvectors (arrays))

        masks = []

        for i in range(len(eigenvectors)):
            mask = eigenvectors[i]
            mask = cv2.resize(mask, (224,224), interpolation=cv2.INTER_NEAREST)
            _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            mask = np.where(mask > 0, 1, 0).astype(np.uint8)

            masks.append({"segmentation": mask, "area": np.sum(mask)})

        masks = sorted(masks, key=lambda x: x["area"], reverse=True) # sort by area
        return masks

def postprocess_masks(masks, threshold=0.5):
    # invert mask to have an area less than 0.5 of the image
    for mask in masks:
        if mask["area"] > threshold * mask["segmentation"].shape[0] * mask["segmentation"].shape[1]:
            mask["segmentation"] = 1 - mask["segmentation"]
            mask["area"] = np.sum(mask["segmentation"])



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

    lost_deg_seed = Lost(alpha=1., k=100, model=dino_model)
    lost_atn_seed = Lost(alpha=0., k=100, model=dino_model)
    
    spectral = DeepSpectralMethods(model=dino_model, n_eigenvectors=5)

    ######################################################################

    setup_time = time.time()

    for ind in tqdm(range(n)):
        
        root = "/nasbrain/datasets/imagenet/images/val/"
        folder = np.random.choice(os.listdir(root))
        path = os.path.join(root,folder)

        img = Image.open(os.path.join(path,np.random.choice(os.listdir(path)))).convert("RGB")
        img = T.Resize((224,224), antialias=True)(img)

        mask_id = identity(img)
        mask_lost = lost_deg_seed(img) + lost_atn_seed(img) # concatenate the output of the two lost blocks
        approx_area = np.mean([mask["area"] for mask in mask_lost]) # average area of the lost masks

        # filter out masks that are too small
        mask_sam = sam(img)
        mask_sam_f = [mask for mask in mask_sam if mask["area"] > 0.4*approx_area] 
        # keep the same magnitude of area as the lost masks
        if len(mask_sam_f) > 2:
            mask_sam = mask_sam_f

        mask_spectral = spectral(img)
        
        # intuitively, the best masks are the ones that are the most similar to the lost masks
        
        # loss = lambda x,y : -iou(x,y)
        # loss = focal_loss
        # loss = dice_loss
        coeff = 1e-13 # black magic
        loss = lambda x,y : dice_loss(x,y) + coeff * focal_loss(x,y)
        # similar loss as the one used in the trainin of SAM
        
        L = [
            loss(mask_lost[0]["segmentation"], mask_sam[i]["segmentation"]) \
            + loss(mask_lost[1]["segmentation"], mask_sam[i]["segmentation"]) \
            for i in range(len(mask_sam))
        ]
        for i in range(len(mask_sam)):
            mask_sam[i]["loss"] = L[i]
            
        mask_sam = [mask_sam[i] for i in np.argsort(L)] # minimize the loss
        
    
        # best masks are the ones that are the most similar to the lost masks
        L = [
            loss(mask_lost[0]["segmentation"], mask_spectral[i]["segmentation"]) \
            + loss(mask_lost[1]["segmentation"], mask_spectral[i]["segmentation"]) \
            for i in range(len(mask_spectral))
        ]

        for i in range(len(mask_spectral)):
            mask_spectral[i]["loss"] = L[i]
        mask_spectral = [mask_spectral[i] for i in np.argsort(L)]
            
        masks = [mask_id, mask_lost, mask_sam, mask_spectral]
        titles = ["Id", "Lost", "SAM", "DSM"]

        # postprocess the masks to have an area less than 0.5 of the image
        postprocess_masks(mask_spectral, threshold=0.5)

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



