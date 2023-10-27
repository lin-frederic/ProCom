import torch

import matplotlib.pyplot as plt
from dataset import EpisodicSampler, FolderExplorer
from model import get_model, forward_dino_v1
from config import cfg
from PIL import Image
import torch
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import lil_matrix
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn
import cv2
from tqdm import tqdm
def knn_affinity(image,k=10):
    # input image is a PIL image
    # first convert rgb to (cos(hue),sin(hue),saturation,value,x,y)
    # then compute the affinity matrix using knn
    # return the affinity matrix
    
    image = image.convert("HSV")

    image = transforms.ToTensor()(image) #(3,h,w)
    cos = torch.cos(image[0]*2*np.pi) #(h,w)
    sin = torch.sin(image[0]*2*np.pi) #(h,w)
    s = image[1] #(h,w)
    v = image[2] #(h,w)
    rows,cols = torch.meshgrid(torch.arange(image.shape[1]),torch.arange(image.shape[2]))
    feature = torch.stack([cos,sin,s,v,rows,cols],dim=2) #(h,w,6)
    feature = feature.reshape(-1,6) #(h*w,6)    
    kdtree = KDTree(feature)
    adjacency_matrix = torch.zeros(feature.shape[0],feature.shape[0])
    for i in tqdm(range(feature.shape[0])):
        _,indices = kdtree.query(feature[i],k) #indices is (k,)
        for index in indices:
            adjacency_matrix[i,index] = 1-torch.norm(feature[i]-feature[index])
        for index in indices:
            adjacency_matrix[index,i] = adjacency_matrix[i,index]
    return adjacency_matrix

def deepSpectralMethods(image_paths,model,device,h=224,w=224):
    # input image_paths is a list of image paths
    # input model is dinov1, should be in eval mode on device
    # output is the eigenvalues and eigenvectors of the normalized laplacian matrix
    image_dict = {}
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((h,w))
        image_tensor = transforms.ToTensor()(image)
        image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])(image_tensor)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)
        h_map, w_map = image_tensor.shape[2]//16, image_tensor.shape[3]//16
        with torch.inference_mode():
            attentions = forward_dino_v1(model,image_tensor).squeeze(0)
        attentions = attentions[1:] #remove cls token, shape is (h_featmap*w_featmap, D)
        attentions = attentions.permute(1,0) # (D,h_featmap*w_featmap)
        attentions = attentions.reshape(attentions.shape[0],h_map,w_map).unsqueeze(0) # (1,D,h_featmap,w_featmap)
        attentions = nn.functional.interpolate(attentions,size=(2*h_map,2*w_map),mode="bilinear")
        attentions = attentions.squeeze(0).reshape(attentions.shape[1],-1)
        attentions = attentions.permute(1,0) # (2*h_featmap*2*w_featmap,D)
        feature_similarity = (attentions @ attentions.T)/(torch.norm(attentions,dim=1).unsqueeze(1) @ torch.norm(attentions,dim=1).unsqueeze(0))
        
        # keep only the positive values
        feature_similarity = feature_similarity * (feature_similarity>0)
        D = torch.diag(torch.sum(feature_similarity,dim=1))
        L = D - feature_similarity # L is (2*h_featmap*2*w_featmap,2*h_featmap*2*w_featmap)
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        # do not keep the first eigenvalue and eigenvector because they are constant
        eigenvalues, eigenvectors = eigenvalues[1:], eigenvectors[:,1:] 
        # the eigenvectors basis might have opposite orientation, so we need to flip the eigenvectors
        for i in range(eigenvectors.shape[1]):
            if eigenvectors[:,i].median() > 0 : # intuition : if the median is positive, then attention is more on the background so we flip the eigenvector
                eigenvectors[:,i] = -eigenvectors[:,i]
        image_dict[image_path] = {"eigenvalues": eigenvalues,
                                "eigenvectors": eigenvectors}
    return image_dict

if __name__ == "__main__":
    folder_explorer = FolderExplorer(cfg.paths)
    paths = folder_explorer()
    sampler = EpisodicSampler(paths = paths,
                            n_query= cfg.sampler.n_queries,
                            n_ways = cfg.sampler.n_ways,
                            n_shot = cfg.sampler.n_shots)
    episode = sampler()
    imagenet_sample = episode["imagenet"]
    class_0 = list(imagenet_sample.keys())[0]
    model = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    image_paths = imagenet_sample[class_0]["support"]
    image_dict = deepSpectralMethods(image_paths,model,device)
    for image_path in tqdm(image_paths): # for each image, plot the first 4 eigenvectors side by side and save the plot
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224,224))
        h,w = image.size
        h_map, w_map = h//16, w//16
        eigenvalues = image_dict[image_path]["eigenvalues"]
        eigenvectors = image_dict[image_path]["eigenvectors"]
        eigenvalues = eigenvalues[:4]
        eigenvectors = eigenvectors[:,:4]
        eigenvectors = eigenvectors.reshape(2*h_map,2*w_map,4) # (2*h_map,2*w_map,4)
        eigenvectors = eigenvectors.permute(2,0,1) # (4,2*h_map,2*w_map)
        eigenvectors = eigenvectors.detach().cpu().numpy()
        eigenvectors = (eigenvectors-eigenvectors.min())/(eigenvectors.max()-eigenvectors.min())
        eigenvectors = (eigenvectors*255).astype(np.uint8)
        plt.figure(figsize=(20,20))
        plt.subplot(1,5,1)
        plt.imshow(image)
        plt.title("original image")
        for i in range(4):
            plt.subplot(1,5,i+2)
            plt.imshow(eigenvectors[i])
            plt.title("eigenvector "+str(i+1))
        image_name = image_path.split("/")[-1].split(".")[0]
        plt.savefig("temp/"+image_name+".png")
    print("done")

        
        