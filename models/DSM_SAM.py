"""
Prompt SAM with coarse masks from DSM to segment images.
-> Hierarchical segmentation
"""

import sys
import os
from typing import Any
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if path not in sys.path:sys.path.append(path)

from models.deepSpectralMethods import DSM
from segment_anything import SamPredictor
from model import get_sam_model

import torch

from torchvision import transforms as T
from tools import ResizeModulo, iou_torch, dice_torch
from PIL import Image
import cv2

from config import cfg
from dataset import DatasetBuilder

import matplotlib.pyplot as plt
import numpy as np 

class DSM_SAM():
    def __init__(self, dsm_model: DSM, sam_model: SamPredictor, nms_thr=0.5):
        super().__init__()
        self.dsm_model = dsm_model
        self.sam_predictor = sam_model
        self.transforms = T.Compose(
            [T.ToTensor(),  
             T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

        self.nms_thr = nms_thr

    def get_metric(self, ref_mask, pred_mask, metric="iou"):
        assert len(ref_mask.shape) == 2, "unbatch ref_mask"
        assert len(pred_mask.shape) == 2, "unbatch pred_mask"

        if abs(ref_mask.shape[0]/ref_mask.shape[1] - pred_mask.shape[0]/pred_mask.shape[1]) > 0.1:
            raise ValueError("ref_mask and pred_mask have different aspect ratios")
        
        
        if ref_mask.shape != pred_mask.shape:
            print("ref_mask and pred_mask have different shapes, resizing ref_mask to pred_mask shape")
            ref_mask = cv2.resize(ref_mask, pred_mask.shape[::-1]) # (W, H) -> (H, W)
        
        if metric == "iou":
            return iou_torch(ref_mask, pred_mask)
        elif metric == "dice":
            return 1-dice_torch(ref_mask, pred_mask)    
        else:
            raise NotImplementedError(f"Metric {metric} is not implemented")
        

    def nms(self, masks, qualities, threshold=0.5, metric="iou"):
        # masks: (n_masks, H, W)
        # qualities: (n_masks)
        # threshold: threshold for NMS
        # metric: metric to use for NMS

        # sort masks by quality
        sorted_idx = torch.argsort(qualities, descending=True)

        # keep the best mask
        kept_masks = [masks[sorted_idx[0]]]
        kept_idx = [sorted_idx[0]]

        sorted_idx = sorted_idx[1:]

        for i in sorted_idx:
            mask = masks[i]
            # compute metrics
            ious = [self.get_metric(kept_mask, mask, metric=metric) for kept_mask in kept_masks]
            if all([iou < threshold for iou in ious]):
                kept_masks.append(mask)
                kept_idx.append(i)
        
        return torch.stack(kept_masks), torch.Tensor(kept_idx).long()
        

    def forward(self, img, sample_per_map=10, temperature=255*0.1):
        # img is a PIL image

        # prepare for DSM
        img_tensor = self.transforms(img)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to("cuda")

        # compute eigen maps (will also be used as coarse masks)
        eigen_maps = self.dsm_model.set_map(img_tensor) # returns numpy array (n_eigen_maps, H, W)

        # compute embeddings for the resized image
        self.sam_predictor.set_image(np.array(img))

        # sample points from eigen maps
        sample_points = self.dsm_model.sample_from_maps(sample_per_map=sample_per_map, temperature=temperature) # (n_eigen_maps, n_samples, 2)

        sample_points = sample_points.reshape(-1, 2) # (n_eigen_maps * n_samples, 2)

        sample_points = torch.from_numpy(sample_points).unsqueeze(1).to("cuda") # (n_eigen_maps * n_samples, 1, 2)

        w, h = img.size
        tranformed_sample_points = self.sam_predictor.transform.apply_coords_torch(sample_points, original_size=(h,w)) # (n_eigen_maps * n_samples, 1, 2)
        points_labels = torch.ones(sample_points.shape[0]).unsqueeze(1).to("cuda") # (n_eigen_maps * n_samples, 1)

        # predict masks from sampled points
        masks, qualities, _ = self.sam_predictor.predict_torch(point_coords=tranformed_sample_points,
                                                                           point_labels=points_labels,
                                                                           multimask_output=True,)
        # -> multimask_output sets the number of masks to 3 (3 granularity levels)
        # masks: (n_eigen_maps * n_samples, 3, H, W)
        # qualities: (n_eigen_maps * n_samples, 3)

        # filter the best mask among the 3 sizes
        kept_masks = []
        kept_qualities = []

        for i, (trimask, triquality) in enumerate(zip(masks, qualities)):
            # trimask: (3, H, W)
            idx = i//sample_per_map # eigen map index
            coarse_ref_mask = torch.from_numpy(eigen_maps[idx]).to("cuda") # (H, W)

            # compute metrics

            metrics = [self.get_metric(coarse_ref_mask, 
                                    mask, metric="iou") for mask in trimask]


            # keep the best one
            best_idx = torch.argmax(torch.Tensor(metrics)).item()
            kept_masks.append(trimask[best_idx])
            kept_qualities.append(triquality[best_idx])

        # NMS on the best masks
        kept_masks = torch.stack(kept_masks)
        kept_qualities = torch.stack(kept_qualities)            
        
        final_masks, final_indexes = self.nms(kept_masks, kept_qualities, threshold=self.nms_thr, metric="iou")
        final_prompts = sample_points[final_indexes]

        return final_masks, final_prompts

        
    
    def __call__(self, img, sample_per_map=10, temperature=255*0.1):
        return self.forward(img, sample_per_map, temperature)
    

def main():
    dsm_model = DSM(n_eigenvectors=5)
    dsm_model.to("cuda")

    sam = get_sam_model(size="b").to("cuda")

    sam_model = SamPredictor(sam)
    
    model = DSM_SAM(dsm_model, sam_model, nms_thr=0.4)

    support_images = ["images/manchot_banane_small.png"]


    for img_name in support_images:
        img = Image.open(img_name).convert("RGB")
        # resize to a multiple of 16
        resized_img = ResizeModulo(patch_size=16, target_size=224, tensor_out=False)(img)   

        masks,points = model(resized_img, sample_per_map=10, temperature=255*0.1)

        lim = min(10, len(masks))
        fig, ax = plt.subplots(1, lim, )
        for i, mask in enumerate(masks[:lim]):
            ax[i].imshow(mask.detach().cpu().numpy())
            ax[i].scatter(points[i,0,0].item(), points[i,0,1].item(), marker="x", color="red")
            ax[i].axis("off")

        plt.tight_layout()

        plt.savefig("temp/DSM_SAM.png")


if __name__ == "__main__":
    main()
    print("Done!")
