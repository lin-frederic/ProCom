import torch

import matplotlib.pyplot as plt
from dataset import EpisodicSampler, FolderExplorer
from model import get_model, forward_dino_v1
from config import cfg
from PIL import Image
import torch
import numpy as np
import scipy
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn
import cv2
from tqdm import tqdm

def knn_affinity(image, n_neighbors=[20, 10], distance_weights=[2.0, 0.1]): # magie noire
    """Computes a KNN-based affinity matrix. Note that this function requires pymatting"""
    try:
        from pymatting.util.kdtree import knn
    except:
        raise ImportError(
            'Please install pymatting to compute KNN affinity matrices:\n'
            'pip3 install pymatting'
        )
    image = np.array(image)/255.0
    h, w = image.shape[:2]
    r, g, b = image.reshape(-1, 3).T
    n = w * h

    x = np.tile(np.linspace(0, 1, w), h) 
    y = np.repeat(np.linspace(0, 1, h), w)

    i, j = [], []

    for k, distance_weight in zip(n_neighbors, distance_weights):
        f = np.stack(
            [r, g, b, distance_weight * x, distance_weight * y],
            axis=1,
            out=np.zeros((n, 5), dtype=np.float32),
        )

        distances, neighbors = knn(f, f, k=k)

        i.append(np.repeat(np.arange(n), k))
        j.append(neighbors.flatten())

    ij = np.concatenate(i + j)
    ji = np.concatenate(j + i)
    coo_data = np.ones(2 * sum(n_neighbors) * n)

    # This is our affinity matrix
    W = scipy.sparse.csr_matrix((coo_data, (ij, ji)), (n, n))
    W = torch.tensor(W.todense(),dtype=torch.float32)
    return W

class DSM(nn.Module):
    def __init__(self, model=None, n_eigenvectors=5,lambda_color=10,device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        if model is None:
            self.model = get_model(size="s",use_v2=False)
        else:
            self.model = model
        self.n = n_eigenvectors
        self.lambda_color = lambda_color
        self.device = device


    def forward(self, img):
        w, h = img.shape[2], img.shape[3]
        w_map, h_map = img.shape[2]//16, img.shape[3]//16
        with torch.inference_mode():
            attentions = forward_dino_v1(self.model,img).squeeze(0)
        attentions = attentions[1:] #remove cls token, shape is (h_featmap*w_featmap, D)
        attentions = attentions.permute(1,0) # (D,h_featmap*w_featmap)
        attentions = attentions.reshape(attentions.shape[0],h_map,w_map).unsqueeze(0) # (1,D,h_featmap,w_featmap)
        attentions = nn.functional.interpolate(attentions,size=(2*h_map,2*w_map),mode="bilinear")
        attentions = attentions.squeeze(0).reshape(attentions.shape[1],-1)
        attentions = attentions.permute(1,0) # (2*h_featmap*2*w_featmap,D)
        feature_similarity = (attentions @ attentions.T)/(torch.norm(attentions,dim=1).unsqueeze(1) @ torch.norm(attentions,dim=1).unsqueeze(0))
        
        # keep only the positive values
        feature_similarity = feature_similarity * (feature_similarity>0)
        #downscale the image to calculate the color affinity matrix, should be (2*h_featmap*2*w_featmap,2*h_featmap*2*w_featmap)
        img = nn.functional.interpolate(img,size=(2*h_map,2*w_map),mode="bilinear")
        img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        img = ((img-img.min())/(img.max()-img.min()))*255
        img = img.astype(np.uint8) 
        color_affinity = knn_affinity(img) # is a numpy array
        #color_affinity = torch.tensor(color_affinity,dtype=torch.float32)
        color_affinity = color_affinity.to(self.device)
        similarity = feature_similarity + self.lambda_color*color_affinity
        D = torch.diag(torch.sum(similarity,dim=1))
        # do not normalize the laplacian matrix because the eigenvalues are very small
        L = D - similarity # L is (2*h_featmap*2*w_featmap,2*h_featmap*2*w_featmap)
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        # do not keep the first eigenvalue and eigenvector because they are constant
        eigenvalues, eigenvectors = eigenvalues[1:], eigenvectors[:,1:] 
        # the eigenvectors basis might have opposite orientation, so we need to flip the eigenvectors
        for i in range(eigenvectors.shape[1]):
            # flip if the median is more than the center of the value range
            if torch.median(eigenvectors[:,i]) > (eigenvectors[:,i].max()+eigenvectors[:,i].min())/2:
                eigenvectors[:,i] = -eigenvectors[:,i]
        
        eigenvectors = eigenvectors[:,:self.n]
        eigenvectors = eigenvectors.reshape(2*h_map,2*w_map,self.n) # (2*h_map,2*w_map,self.n)
        eigenvectors = eigenvectors.permute(2,0,1) # (self.n,2*h_map,2*w_map)
        eigenvectors = eigenvectors.detach().cpu().numpy()

        temp = []
        for vector in eigenvectors:
            vector = ((vector-vector.min())/(vector.max()-vector.min()))*255 # normalize between 0 and 255 to have a grayscale image
         
            temp.append(cv2.resize(vector.astype(np.uint8), 
                                            (w,h), 
                                            interpolation=cv2.INTER_NEAREST)) # resize to original image size
            


        eigenvectors = np.stack(temp,axis=0) # (self.n,h,w)

        return eigenvectors
     
        
        


if __name__ == "__main__":
    folder_explorer = FolderExplorer(cfg.paths)
    paths = folder_explorer()
    sampler = EpisodicSampler(paths = paths,
                            n_query= cfg.sampler.n_queries,
                            n_ways = cfg.sampler.n_ways,
                            n_shot = cfg.sampler.n_shots)
    dsm = DSM() # default parameters are model=None, n_eigenvectors=5
    dsm.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dsm.to(device)
    # dataset
    episode = sampler()
    imagenet_sample = episode["imagenet"]
    # support set
    support_images = [image_path for classe in imagenet_sample for image_path in imagenet_sample[classe]["support"]] # default n_shot=1, n_ways=5
    for image_path in tqdm(support_images):
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224,224))
        image_tensor = transforms.ToTensor()(image)
        image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])(image_tensor)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)
        eigenvectors = dsm(image_tensor)
        # save the plot of the eigenvectors side by side for each image and save it
        fig, axs = plt.subplots(1, eigenvectors.shape[0], figsize=(15,15))
        for i,ax in enumerate(axs):
            ax.imshow(eigenvectors[i],cmap="viridis")
            ax.axis("off")
        plt.savefig(f"temp/eigenvectors_{image_path.split('/')[-1]}")
        plt.close()
        #also save the plot of the original image
        plt.imshow(image)
        plt.axis("off")
        plt.savefig(f"temp/original_{image_path.split('/')[-1]}")
        plt.close()
        