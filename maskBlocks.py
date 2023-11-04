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

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, img):
        w, h = img.size
        return [{"segmentation": np.ones((w,h),dtype=np.uint8), "area": w*h}] # whole image

class Lost(nn.Module):
    def __init__(self,alpha, k):
        super().__init__()
        self.res = 224
        self.up = 2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = get_model(size="s",use_v2=False) # loads a DINOv1 model, size s
        model.to(self.device)
        self.lost = Lost_module(model=model,alpha=alpha,k=k*self.up**2)

    def clean(self, out, w, h):
        out = cv2.resize(out.astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST)

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

        return [{"segmentation": mask,                          "area": np.sum(mask)},
                {"segmentation": np.ones((w,h),dtype=np.uint8), "area": w*h}]

class SAM(nn.Module):
    def __init__(self, size="b"):
        super().__init__()
        sizes = {
            "b" : "sam_vit_b_01ec64.pth",
            "l" : "sam_vit_l_0b3195.pth",
            "h" : "sam_vit_h_4b8939.pth",
        }

        sam = sam_model_registry[f"vit_{size}"](checkpoint=f"/nasbrain/f21lin/{sizes[size]}")
        print("SAM loaded")
        sam.to("cuda")

        self.AMG = SamAutomaticMaskGenerator(sam)
        
    def forward(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        masks = self.AMG.generate(img)
        return [{"segmentation": mask["segmentation"], 
                 "area": mask["area"]} for mask in masks]
        
class DeepSpectralMethods(nn.Module):
    def __init__(self, model_size="s", n_eigenvectors=5):
        super().__init__()
        self. transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
        ])
        self.dsm = DSM(model_size=model_size, n_eigenvectors=n_eigenvectors)
    def forward(self, img):
        
        img = self.transforms(img).unsqueeze(0).to("cuda")
        eigenvectors = self.dsm(img) # returns a list of eigenvectors (arrays))

        masks = []

        for i in range(len(eigenvectors)):
            mask = eigenvectors[i]
            _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            masks.append({"segmentation": mask, "area": np.sum(mask)})

        masks = sorted(masks, key=lambda x: x["area"], reverse=True) # sort by area
        return masks

        

import os
import matplotlib.pyplot as plt
def main():
    np.random.seed(1)

    root = "/nasbrain/datasets/imagenet/images/val/"
    folder = np.random.choice(os.listdir(root))
    path = os.path.join(root,folder)

    img = Image.open(os.path.join(path,np.random.choice(os.listdir(path)))).convert("RGB")
    img = T.Resize((224,224), antialias=True)(img)

    identity = Identity()
    lost = Lost(alpha=0., k=100)
    sam = SAM(size="b")
    spectral = DeepSpectralMethods(model_size="s", n_eigenvectors=5)

    mask_id = identity(img)
    mask_lost = lost(img)
    mask_sam = sam(img) 
    mask_spectral = spectral(img)


    masks = [mask_id, mask_lost, mask_sam, mask_spectral]
    titles = ["Identity", "Lost", "SAM", "DSM"]

    n = 5
    fig, axes = plt.subplots(len(masks)+1, n, figsize=(10,10))
    axes[0,0].imshow(img)
    axes[0,0].set_title("Original image")
    axes[0,0].axis("off")

    for i, (mask, title) in enumerate(zip(masks, titles)):
        for j in range(len(mask))[:n]:
            axes[i+1,j].imshow(mask[j]["segmentation"])
            axes[i+1,j].set_title(f"{title} - area : {mask[j]['area']}")
            
    for i in range(len(masks)+1):
        for j in range(n):
            axes[i,j].axis("off")


    plt.tight_layout()

    plt.savefig("temp/img.png")

    print("done")

if __name__ == "__main__":
    main()



