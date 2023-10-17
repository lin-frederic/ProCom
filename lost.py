# Adapted from https://arxiv.org/pdf/2109.14279.pdf (Section 3.2)

import torch
from torch import nn
from model import get_model, forward_dino_v1
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

class Lost(nn.Module):
    def __init__(self, model, k = 100):
        super().__init__()
        self.k = k # indicates the cardinality of the seed expansion set
        self.model = model # DINO model
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
        # compute similarity matrix, degree matrix
        similarity_matrix = torch.matmul(out, out.T)              # (H_d*W_d, H_d*W_d)

        degree_matrix = similarity_matrix>=0                      # (H_d*W_d, H_d*W_d)
        # select seed with lowest degree
        seed = torch.argmin(degree_matrix.sum(dim=0)) # or dim = 1, doesn't matter 
        # (without loss of generality, we make a choice here)

        # expand seed set on similarity matrix
        degree_matrix[seed][seed] = 255
        set_seed = degree_matrix[seed].reshape(H_d,W_d).detach().cpu().numpy()
        
        set_seed = np.uint8(set_seed)
        
        # TODO : limit cardinality of seed expansion set to k
        # TODO : apply the box extraction algorithm
        # reminder : the box extraction algorithm is 
        # 1 if the sum of the dot product of the vector and all the vectors in the set is greater than 0
        # 0 otherwise
        
        return set_seed


def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(size="s",use_v2=False) # loads a DINOv1 model, size s
    model.to(device)
    
    lost = Lost(model)
    
    path = "/nasbrain/datasets/imagenet/images/val/n01514668/ILSVRC2012_val_00000911.JPEG"
    path = "/nasbrain/datasets/imagenet/images/val/n01514668/ILSVRC2012_val_00004463.JPEG"
    img = Image.open(path)
    img.save("temp/img.png")
    
    w, h = img.size
    up = 2
    w, h = w*up, h*up
    img = T.Resize((h//16*16,w//16*16), antialias=True)(img)
    img = T.ToTensor()(img).unsqueeze(0).to(device)
    
    mask = lost(img)
    
    # save mask
    
    plt.imsave("temp/mask.png", mask)
    print("Done")

if __name__ == "__main__":
    main()