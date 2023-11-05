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

def deepSpectralMethods(image_paths,model,device,h=224,w=224,lambda_color=10):
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
        #downscale the image to calculate the color affinity matrix, should be (2*h_featmap*2*w_featmap,2*h_featmap*2*w_featmap)
        image = image.resize((2*h_map,2*w_map))
        affinity_matrix = knn_affinity(image)
        affinity_matrix = affinity_matrix.to(device)
        similarity = feature_similarity + lambda_color*affinity_matrix
        D = torch.diag(torch.sum(similarity,dim=1))
        L = D - similarity # L is (2*h_featmap*2*w_featmap,2*h_featmap*2*w_featmap)
        #L = D**(-1/2) @ L @ D**(-1/2) # normalized laplacian matrix, often D have very great values so the normalized laplacian matrix is not well conditioned as eigenvalues are very small
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        # do not keep the first eigenvalue and eigenvector because they are constant
        eigenvalues, eigenvectors = eigenvalues[1:], eigenvectors[:,1:] 
        # the eigenvectors basis might have opposite orientation, so we need to flip the eigenvectors
        for i in range(eigenvectors.shape[1]):
            # flip if the median is more than the center of the value range
            if torch.median(eigenvectors[:,i]) > (eigenvectors[:,i].max()+eigenvectors[:,i].min())/2:
                eigenvectors[:,i] = -eigenvectors[:,i]
        image_dict[image_path] = {"eigenvalues": eigenvalues,
                                "eigenvectors": eigenvectors}
    return image_dict

class DSM(nn.Module):
    def __init__(self, model=None, n_eigenvectors=5):
        super().__init__()
        if model is None:
            self.model = get_model(model_size="s",use_v2=False)
        else:
            self.model = model
        self.n = n_eigenvectors


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
        D = torch.diag(torch.sum(feature_similarity,dim=1))
        L = D - feature_similarity # L is (2*h_featmap*2*w_featmap,2*h_featmap*2*w_featmap)
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        # do not keep the first eigenvalue and eigenvector because they are constant
        eigenvalues, eigenvectors = eigenvalues[1:], eigenvectors[:,1:] 
        # the eigenvectors basis might have opposite orientation, so we need to flip the eigenvectors
        for i in range(eigenvectors.shape[1]):
            # flip if the median is more than the center of the value range
            if torch.median(eigenvectors[:,i]) > (eigenvectors[:,i].max()+eigenvectors[:,i].min())/2:
                eigenvectors[:,i] = -eigenvectors[:,i]
        
        eigenvectors = eigenvectors[:,:self.n]
        eigenvectors = eigenvectors.reshape(2*h_map,2*w_map,self.n) # (2*h_map,2*w_map,4)
        eigenvectors = eigenvectors.permute(2,0,1) # (self.n,2*h_map,2*w_map)
        eigenvectors = eigenvectors.detach().cpu().numpy()

        temp = []
        for vector in eigenvectors:
            vector = ((vector-vector.min())/(vector.max()-vector.min()))*255
         
            temp.append(cv2.resize(vector.astype(np.uint8), 
                                            (w,h), 
                                            interpolation=cv2.INTER_NEAREST))
            


        eigenvectors = np.stack(temp,axis=0)

        return eigenvectors


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
    lambda_color = 10
    image_dict = deepSpectralMethods(image_paths,model,device,lambda_color=lambda_color)
    for image_path in tqdm(image_paths): # for each image, plot the first 4 eigenvectors side by side and save the plot
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224,224))
        h,w = image.size
        h_map, w_map = h//16, w//16
        eigenvalues = image_dict[image_path]["eigenvalues"]
        eigenvectors = image_dict[image_path]["eigenvectors"] # (2*h_map*2*w_map,2*h_map*2*w_map)
        x,y,w,h,fiedler = objectLocalization(eigenvectors,h,w)
        eigenvalues = eigenvalues[:4]
        eigenvectors = eigenvectors[:,:4] # (2*h_map*2*w_map,4)
        eigenvectors = eigenvectors.reshape(2*h_map,2*w_map,4) # (2*h_map,2*w_map,4)
        eigenvectors = eigenvectors.permute(2,0,1) # (4,2*h_map,2*w_map)
        eigenvectors = eigenvectors.detach().cpu().numpy()
        eigenvectors = (eigenvectors-eigenvectors.min())/(eigenvectors.max()-eigenvectors.min())
        eigenvectors = (eigenvectors*255).astype(np.uint8)
        plt.figure(figsize=(20,10))
        plt.subplot(1,5,1)
        plt.imshow(image)
        plt.title("original image")
        for i in range(4):
            plt.subplot(1,5,i+2)
            plt.imshow(eigenvectors[i])
            plt.title("eigenvector "+str(i+1))
        image_name = image_path.split("/")[-1].split(".")[0]
        plt.savefig("temp/"+image_name+"_"+str(lambda_color)+".png")
        plt.figure(figsize=(20,20)) # plot the object localization against binary segmentation
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.title("object localization")
        plt.axis("off")
        from matplotlib.patches import Rectangle
        plt.gca().add_patch(Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none'))
        plt.subplot(1,2,2)
        plt.imshow(fiedler)
        plt.title("binary segmentation")
        plt.axis("off")
        plt.savefig("temp/"+image_name+"_"+str(lambda_color)+"_localization.png")
    print("done")

        
        