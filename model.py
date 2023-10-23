import torch
from torch import nn
import cv2
from PIL import Image
import torchvision.transforms as T
import numpy as np
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt


load_refs = {
    "s":"dinov2_vits14",
    "b":"dinov2_vitb14",
    "l":"dinov2_vitl14",
    "g":"dinov2_vitg14"
}

repo_ref = "facebookresearch/dinov2"


def get_model(size="s",use_v2=False):
    if use_v2:
        if size == "s":
            model = torch.hub.load(repo_ref, load_refs[size])
        elif size == "b":
            model = torch.hub.load(repo_ref, load_refs[size])
        elif size == "l":
            model = torch.hub.load(repo_ref, load_refs[size])
        elif size == "g":
            model = torch.hub.load(repo_ref, load_refs[size])
    else:
        if size == "s":
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        elif size == "b":
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        elif size == "l":
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vitl16')
    return model

# adapted from the official repos
def forward_dino_v1(model, x):
    x = model.prepare_tokens(x)
    for blk in model.blocks:
        x = blk(x)
    x = model.norm(x)
    return x

# for dinov2, pass the is_training = True flag


def show_attn(model, x, is_v2=False):
    if is_v2:
        B, C, H, W = x.shape
        w_featmap = W // 14
        h_featmap = H // 14
        with torch.inference_mode():
            output = model.get_intermediate_layers(x = x,
                                                   reshape = True,
                                                   n = 2,
                                                   return_class_token = True,)
            maps = output[0][0] 
            B, C = output[0][1].shape
            
            # reshape maps to be (B, N, C) where N is the number of patches
            maps = maps.reshape((B,maps.shape[1],-1)).permute(0,2,1)
            class_token = output[0][1].reshape((B,-1,1)).permute(0,2,1)
            maps = torch.cat((class_token, maps), dim=1)
            # get the last attention block (only qkv)with 
            qkv = model.blocks[-1].attn.qkv
            B, N, C = maps.shape
            qkv_out = qkv(maps).reshape(B, N, 3, model.num_heads, C // model.num_heads).permute(2, 0, 3, 1, 4) # (3, B, num_heads, N, C//num_heads)
            head_dim = C // model.num_heads
            scale = head_dim**-0.5
            q, k = qkv_out[0] * scale, qkv_out[1]
            attn = q @ k.transpose(-2, -1) # (B, nh, N, N)
            nh = model.num_heads
            assert B == 1, "B must be 1"
            attn = attn[:, :, 0, 1:].reshape(B,nh, h_featmap, w_featmap)
            return attn
        
    else: # dinov1
        B, C, H, W = x.shape
        w_featmap = W // 16
        h_featmap = H // 16
        attn_map = model.get_last_selfattention(x)
        attn_map = attn_map[:, :, 0, 1:].reshape(B, attn_map.shape[1], h_featmap, w_featmap)
        return attn_map
        
def get_seed_from_attn(attn_map):
    # attn_map is (B, nh, H, W)
    # size is (H, W) or S
    
    array_map = torch.min(attn_map, dim=1)[0].squeeze().detach().cpu().numpy()
    

    
    array_map = (array_map - array_map.min()) / (array_map.max() - array_map.min())
    array_map = (255*array_map).astype(np.uint8)
    
    _, array_map = cv2.threshold(array_map, 200, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    
    barycenter = center_of_mass(array_map)
    
    barycenter = (int(barycenter[0]), int(barycenter[1]))
    
    return torch.Tensor(np.ravel(barycenter))
         
    
    
def main():
    model = get_model(size="s",use_v2=False)
    
    img = Image.open("temp/img_8.png")
    
    img = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
    ])(img).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    img = img.to(device)
    
    with torch.inference_mode():
        attn_map = show_attn(model, img)
        
    array_map = torch.min(attn_map, dim=1)[0].squeeze().detach().cpu().numpy() 
    # min returns (values, indices)
    array_map = cv2.resize(array_map, (224,224))
    array_map = (array_map - array_map.min()) / (array_map.max() - array_map.min())
    array_map = (255*array_map).astype(np.uint8)
    
    _, array_map = cv2.threshold(array_map, 100, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    
    barycenter = center_of_mass(array_map)
    
    
    plt.imshow(array_map, cmap="viridis")
    
    plt.scatter(barycenter[1], barycenter[0], c="red", s=100, marker="x")
    
    plt.colorbar()
    
    plt.savefig("temp/attn.png")
    
    
        
    
if __name__ == "__main__":
    main()