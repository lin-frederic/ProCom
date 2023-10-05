import torch
import torch.nn.functional as F

class preprocess_NCM(torch.nn.Module):
    # preprocess_NCM: preprocess the features for NCM
    # including: 1. subtract the mean feature vector
    #            2. normalize the feature vector to make it into hypersphere
    # input: support_features: [n_way, n_shot, d]
    #        query_features: [n_query, d]
    # output: sphered_support_features: [n_way, n_shot, d]
    #         sphered_query_features: [n_query, d]
    def __init__(self):
        super(preprocess_NCM, self).__init__()
    def forward(self, support_features, query_features):
        mean_features = torch.mean(support_features, dim=[0,1],keepdim=True) # [1, 1, d]
        support_features = support_features - mean_features # [n_way, n_shot, d]
        query_features = query_features - mean_features.squeeze(0) # [n_query, d]
        sphered_support_features = support_features/torch.norm(support_features, dim=2, keepdim=True)
        sphered_query_features = query_features/torch.norm(query_features, dim=1, keepdim=True)
        return sphered_support_features, sphered_query_features
    

class NCM(torch.nn.Module):
    def __init__(self):
        super(NCM, self).__init__()

    def forward(self, support_features, query_features, use_cosine=True):
        # support_features: [n_way, n_shot, d]
        # query_features: [n_query, d]
        # output: [n_way] each element is the predicted class for each query

        # Preprocess the features
        preprocess_NCM_layer = preprocess_NCM()
        support_features, query_features = preprocess_NCM_layer(support_features, query_features)

        if use_cosine: 
            # Reshape query_features for broadcasting
            query_features = query_features.unsqueeze(1)  # [n_query, 1, d] 
            query_features = query_features.unsqueeze(0)        # [1, n_query, 1, d]
            support_features = support_features.unsqueeze(1)    # [n_way, 1, n_shot, d]

            # torch.sum works only if the dimensions are the same or one of them is 1 (broadcasting)
    
            # Calculate cosine similarities
            cosine_similarities = torch.sum(support_features * query_features, dim=3)  # [n_way, n_query, n_shot]

            # Calculate the best similarity for each class
            best_similarities = torch.max(cosine_similarities, dim=2)[0] # [n_way, n_query]

            # Calculate the best class for each query
            output = torch.argmax(best_similarities, dim=0) # [n_way]

            return output
    
        else:
            # use torch.cdist to calculate the euclidean distance

            # calculate the euclidean distance
            euclidean_distance = torch.cdist(query_features, support_features, p=2.0)
            
            # calculate the best distance for each class
            best_distance = torch.min(euclidean_distance, dim=2)[0]

            # calculate the best class for each query
            output = torch.argmin(best_distance, dim=0)

            return output



def test():
    # Test NCM
    ncm = NCM()
    support_features = torch.randn(4, 6, 10)
    query_features = torch.randn(15, 10)
    output = ncm(support_features, query_features, False)
    print("Euclidean distance: ", output)
    output = ncm(support_features, query_features)
    print("Cosine similarity: ", output)


if __name__ == '__main__':
    test()
