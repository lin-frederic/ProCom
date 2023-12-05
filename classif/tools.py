import torch
import torch.nn.functional as F

class preprocess_Features(torch.nn.Module):
    # preprocess_NCM: preprocess the features for NCM
    # including: 1. subtract the mean feature vector
    #            2. normalize the feature vector to make it into hypersphere
    # input: support_features: list of features, as a tensor
    #        query_features: list of features, as a tensor
    # output: support_output: list of features, as a tensor
    #         sphered_query_features: list of features, as a tensor
    def __init__(self):
        super(preprocess_Features, self).__init__()
    def forward(self, support_features, query_features):
        mean_feature = torch.mean(support_features, dim=0) # [d]
        sphered_support_features = support_features - mean_feature
        sphered_support_features = F.normalize(sphered_support_features, p=2, dim=1) # [n_shot, d]
        sphered_query_features = query_features - mean_feature
        sphered_query_features = F.normalize(sphered_query_features, p=2, dim=1) # [n_query, d]
        
        return sphered_support_features, sphered_query_features