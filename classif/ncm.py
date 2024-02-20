import torch
import torch.nn.functional as F
import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if path not in sys.path:sys.path.append(path)
from tools import preprocess_Features
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from config import cfg

class NCM(torch.nn.Module):
    def __init__(self, top_k=1, seed=-1):
        super(NCM, self).__init__()
        self.preprocess_NCM_layer = preprocess_Features()
        self.top_k = top_k
        self.seed = seed

    def forward(self, support_features, query_features, support_labels, query_labels, use_cosine=True):
        # support_features: list of features, as a tensor of shape [Ns,d] where Ns is the number of support features
        # support_labels: list of (class,image_index) tuples
        # query_features: list of features, as a tensor of shape [Nq,d] where Nq is the number of query features
        # output: [n_query,class] each element is the predicted class for each query
        # Preprocess the features
        
        #support_features, query_features = self.preprocess_NCM_layer(support_features, query_features)
        acc = 0
        # calculate similarity between each query feature and each support feature
        if use_cosine:
            similarity = F.cosine_similarity(query_features.unsqueeze(1), support_features.unsqueeze(0), dim=2) # [n_query, n_shot]
        else:
            similarity = -torch.cdist(query_features.unsqueeze(1), support_features.unsqueeze(0), p=2)
        #similarity is (Nq, Ns), Nq != number of query images, Ns != number of support images
        # problem is we can have multiple features per image (as we have multiple masks)
        # for each query, slice the similarity tensor to get the similarity with the support features of the same class
        # then take the max similarity
        similarity = similarity.squeeze(1) # [Nq, Ns]
        # unique_support_labels = {image_index: class_index}
        unique_labels = [] # list of unique class indices
        unique_support_labels = {} # class_index: [image_index]
        unique_support_labels_reverse = {} # image_index: class_index
        unique_query_labels = {} # class_index: [image_index]
        unique_query_labels_reverse = {} # image_index: class_index
        annotations = {} # image_index: [crop_index]
        annotations['support'] = {}
        annotations['query'] = {}
        annotations_reverse = {} # crop_index:image_index
        annotations_reverse["support"] = {}
        annotations_reverse["query"] = {}
        for i, (class_index, image_index) in enumerate(support_labels):
            if class_index not in unique_labels:
                unique_labels.append(class_index)
            if class_index not in unique_support_labels:
                unique_support_labels[class_index] = []
            if image_index not in unique_support_labels[class_index]:
                unique_support_labels[class_index].append(image_index)
            if image_index not in unique_support_labels_reverse:
                unique_support_labels_reverse[image_index] = class_index
            if image_index not in annotations['support']:
                annotations['support'][image_index] = []
            annotations['support'][image_index].append(i)
            if i not in annotations_reverse["support"]:
                annotations_reverse["support"][i] = image_index
        for i, (class_index, image_index) in enumerate(query_labels):
            if class_index not in unique_query_labels:
                unique_query_labels[class_index] = []
            if image_index not in unique_query_labels[class_index]:
                unique_query_labels[class_index].append(image_index)
            if image_index not in unique_query_labels_reverse:
                unique_query_labels_reverse[image_index] = class_index
            if image_index not in annotations['query']:
                annotations['query'][image_index] = []
            annotations['query'][image_index].append(i)
            if i not in annotations_reverse["query"]:
                annotations_reverse["query"][i] = image_index
        # Classify each query feature
        acc = np.zeros(cfg.sampler.n_shots) # accuracy for each shot, to plot the curve
        for image_index in annotations['query']:
            for k in range(cfg.sampler.n_shots):
                # sample k+1 indices from unique_support_labels[class_index]
                sampled_support_indices = {}
                for support_class_index in unique_labels:
                    if self.seed > 0:
                        np.random.seed(self.seed)
                    sampled_support_indices[support_class_index] = np.random.choice(unique_support_labels[support_class_index], k+1, replace=False)
                augmented_support_indices = []
                for support_class_index in unique_labels:
                    for index in sampled_support_indices[support_class_index]:
                        for j in annotations['support'][index]:
                            augmented_support_indices.append(j)
                augmented_query_indices = []
                for j in annotations['query'][image_index]:
                    augmented_query_indices.append(j)
                max_similarity = -float('inf')
                max_similarity_index = -float('inf')
                similarity_sampled = similarity[augmented_query_indices][:,augmented_support_indices]
                for j in range(len(augmented_query_indices)):
                    for l in range(len(augmented_support_indices)):
                        if similarity_sampled[j,l] > max_similarity:
                            max_similarity = similarity_sampled[j,l]
                            max_similarity_index = l
                support_crop_index = augmented_support_indices[max_similarity_index]
                support_image_index = annotations_reverse["support"][support_crop_index]
                support_class = unique_support_labels_reverse[support_image_index]
                query_class = unique_query_labels_reverse[image_index]
                if support_class == query_class:
                    acc[k] += 1
        acc = acc / len(annotations['query'])
        return acc[0]
def preprocess_plot(img):
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype('uint8')
    return img


def test():
    # Test NCM
    ncm = NCM()
    support_features = torch.randn(5*20*3, 10)
    query_features = torch.randn(20*5*3, 10)
    support_labels = [(i//(20*3), i//3) for i in range(5*20*3)]
    query_labels = [(i//(20*3), i//3) for i in range(20*5*3)]
    acc = ncm(support_features, query_features, support_labels, query_labels)
    print(acc)
def test2():
    # Test NCM
    ncm = NCM()
    support_features = torch.randn(5*2*3, 10)
    query_features = torch.randn(15*5*3, 10)
    support_labels = [(i//(2*3), i//3) for i in range(5*2*3)]
    query_labels = [(i//(15*3), i//3) for i in range(15*5*3)]
    support_augmented_imgs = [torch.randn(1,3, 224, 224) for i in range(5*2*3)]
    query_augmented_imgs = [torch.randn(1,3, 224, 224) for i in range(15*5*3)]
    support_augmented_imgs = [img.squeeze(0).permute(1,2,0).cpu().numpy() for img in support_augmented_imgs]
    query_augmented_imgs = [img.squeeze(0).permute(1,2,0).cpu().numpy() for img in query_augmented_imgs]
    to_display = (support_augmented_imgs, query_augmented_imgs)
    acc = ncm(support_features, query_features, support_labels, query_labels, to_display=to_display)
    print(acc)
def test3():
    ncm = NCM()
    support_features = torch.randn(5*20*3, 10)
    query_features = torch.randn(15*5*3, 10)
    support_labels = [(i//(20*3), i//3) for i in range(5*20*3)]
    query_labels = [(i//(15*3), i//3) for i in range(15*5*3)]
    support_augmented_imgs = [torch.randn(1,3, 224, 224) for i in range(5*20*3)]
    query_augmented_imgs = [torch.randn(1,3, 224, 224) for i in range(15*5*3)]
    support_augmented_imgs = [img.squeeze(0).permute(1,2,0).cpu().numpy() for img in support_augmented_imgs]
    query_augmented_imgs = [img.squeeze(0).permute(1,2,0).cpu().numpy() for img in query_augmented_imgs]
    similarity = ncm(support_features, query_features, support_labels, query_labels,calculate_accuracy=False,use_cosine=True)
    
if __name__ == '__main__':
    test()
