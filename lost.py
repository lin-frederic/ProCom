# Adapted from https://arxiv.org/pdf/2109.14279.pdf (Section 3.2)

import torch
from torch import nn
from model import get_model, forward_dino_v1, show_attn, get_seed_from_attn
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

from tqdm import tqdm
import argparse

from tools import unravel_index # don't change to utils, there is a utils.py elsewhere

class Lost(nn.Module):
    def __init__(self, model, alpha, k = 100):
        super().__init__()
        self.k = k # indicates the cardinality of the seed expansion set
        self.model = model # DINO model
        self.alpha = alpha # alpha = 1 means that the seed is the pixel with the lowest degree
                           # alpha = 0 means that the seed is the barycenter of the thresholded (otsu) attention map
        assert 0 <= self.alpha <= 1, "alpha must be between 0 and 1"
        self.model.eval()
        
    def forward(self, img):
        """
        Args:
        img : input image (tensor) of shape (1,3,H,W)
        
        Output:
        mask : mask of shape (1,H_d,W_d) indicating the pixels 
        that are part of the seed expansion set
        H_d and W_d are the dimensions of the patched image
        """
        H_d, W_d = img.shape[2]//16, img.shape[3]//16
        assert img.shape[2] % 16 == 0 and img.shape[3] % 16 == 0, "image dimensions must be divisible by 16"
        # run through the model
        with torch.inference_mode():
            out = forward_dino_v1(self.model, img).squeeze(0) 
            # remove cls token
            out = out[1:] # (H_d*W_d, D)
            attn = show_attn(self.model, img, is_v2=False)
        
        # get attention seed
        attn_seed = get_seed_from_attn(attn) # (2,)
        
        # compute similarity matrix, degree matrix
        similarity_matrix = torch.matmul(out, out.T)              # (H_d*W_d, H_d*W_d)
        

        degree_matrix = similarity_matrix>=0                      # (H_d*W_d, H_d*W_d)
        # select seed with lowest degree
        seed_degree = torch.argmin(degree_matrix.sum(dim=0)) # or dim = 1, doesn't matter 
        # (without loss of generality, we make a choice here)

        
        # the seed is a convex combination of the attention seed and the seed with lowest degree
        
        # unravel coordinates
        seed_degree = unravel_index(seed_degree, (H_d,W_d)) # (2,)
        # attn seed is already in (y,x) format

        # compute seed
        seed_degree = seed_degree.to("cpu") 
        attn_seed = attn_seed.to("cpu") 
        
        seed = self.alpha*seed_degree + (1-self.alpha)*attn_seed # (2,)
        seed = torch.round(seed).type(torch.int64) # (2,)

        
        # convert seed to index
        seed = seed[0]*W_d + seed[1] # (1,)
        
        # expand seed set on similarity matrix
        degree_matrix[seed][seed] = 255
        set_seed = degree_matrix[seed]                            # (H_d*W_d,)

        # limit cardinality of seed expansion set to k
        ordered_set_seed = torch.argsort(similarity_matrix[seed], descending=True) # returns indices
        set_seed[ordered_set_seed[self.k:]] = 0                        # (H_d*W_d,)

        # box extraction algorithm
        for i in range(len(set_seed)):
            if set_seed[i] == 0:
                continue
            else:
                if torch.sum(similarity_matrix[i][set_seed==1]) > 0: 
                # if the sum of the similarities between the current pixel and the pixels in the seed expansion set is > 0
                    set_seed[i] = 1
                else:
                    set_seed[i] = 0

        set_seed = set_seed.reshape(H_d, W_d).detach().cpu().numpy()
        set_seed = np.uint8(set_seed*255)
        # extract the largest connected component
        set_seed = cv2.connectedComponents(set_seed)[1] # index 0 is background

        return set_seed


def main(n, is_grid):
    
    np.random.seed(42)
    
    res = 224
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(size="s",use_v2=False) # loads a DINOv1 model, size s
    model.to(device)
    
    k = 400 if is_grid else 100 # 4 images so 400 patches
    lost = Lost(model, alpha=0.5, k=k)
    
    for index in tqdm(range(n)):
    
        #path = "/nasbrain/datasets/imagenet/images/val/n01514668/ILSVRC2012_val_00000911.JPEG"
        #path = "/nasbrain/datasets/imagenet/images/val/n01514668/ILSVRC2012_val_00004463.JPEG"
        root = "/nasbrain/datasets/imagenet/images/val/"
        folder = np.random.choice(os.listdir(root))
        path = os.path.join(root,folder)
        
        #path = "/nasbrain/datasets/imagenet/images/val/n01514668/"
        #path = "/nasbrain/datasets/ADE20K/ADEChallengeData2016/images/validation/ADE_val_00001399.jpg"
        
        # concatenate 4 images (grid)
        if is_grid:
            blank = Image.new("RGB", (res*2,res*2))
        
            for i,img_path in enumerate(os.listdir(path)[:4]):
                img = Image.open(os.path.join(path,img_path)).convert("RGB")
                #img = T.CenterCrop(res)(img)
                img = T.Resize((res,res), antialias=True)(img)
                blank.paste(img, (res*(i%2), res*(i//2)))
                
            img = blank
        else:
            name = np.random.choice(os.listdir(path))
            img = Image.open(os.path.join(path,name)).convert("RGB")
            #img = T.CenterCrop(res)(img)
            img = T.Resize((res,res), antialias=True)(img)
            
        img.save(f"temp/img_{index}.png")
        
        w, h = img.size
        up = 2 # upscaling factor
        w, h = int(w*up), int(h*up)
        img_t = T.Resize((h//16*16,w//16*16), antialias=True)(img)
        img_t = T.ToTensor()(img_t).unsqueeze(0).to(device)
        
        mask = lost(img_t)
        
        # save mask

        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        plt.imsave(f"temp/mask_{index}.png", mask)
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--grid", action=argparse.BooleanOptionalAction, default=True)
    n = parser.parse_args().n
    is_grid = parser.parse_args().grid
    main(n, is_grid)