import torch
import torch.nn.functional as F
from tools import preprocess_Features

class NCM(torch.nn.Module):
    def __init__(self):
        super(NCM, self).__init__()
        self.preprocess_NCM_layer = preprocess_Features()

    def forward(self, support_features, query_features, support_labels, query_labels, use_cosine=False):
        # support_features: list of features, as a tensor of shape [n_shot, d]
        # support_labels: list of (class,image_index) tuples
        # query_features: list of features, as a tensor of shape [n_query, d]
        # output: [n_query,class] each element is the predicted class for each query
        # Preprocess the features
        
        support_features, query_features = self.preprocess_NCM_layer(support_features, query_features)
        acc = 0
        # calculate similarity between each query feature and each support feature
        if use_cosine:
            similarity = F.cosine_similarity(query_features.unsqueeze(1), support_features.unsqueeze(0), dim=2) # [n_query, n_shot]
        else:
            similarity = -torch.cdist(query_features.unsqueeze(1), support_features.unsqueeze(0), p=2)
        #similarity is (n_query, n_shot)
        # problem is we can have multiple features per image (as we have multiple masks)
        # for each query, slice the similarity tensor to get the similarity with the support features of the same class
        # then take the max similarity
        similarity = similarity.squeeze(1)
        visited = set()
        for i, (query_class, query_index) in enumerate(query_labels):
            #find where the other masks of the same image are
            if query_index not in visited:
                visited.add(query_index)
                indices = [j for j, (class_, index_) in enumerate(query_labels) if index_ == query_index]
                query_similarity = similarity[indices, :] # [n_masks, n_shot]
                support_similarity, support_indices = torch.max(query_similarity, dim=1) # [n_masks]
                # this would be the predicted class for each mask
                # now we take the highest similarity
                mask_sim, mask_index = torch.max(support_similarity, dim=0)
                pred_idx = support_indices[mask_index]
                pred_class = support_labels[pred_idx][0]
                acc += int(pred_class == query_class)
        acc = acc / len(query_features)
        return acc
        



def test():
    # Test NCM
    ncm = NCM()
    support_features = torch.randn(5*2*3, 10)
    query_features = torch.randn(15*5*3, 10)
    support_labels = [(i//(2*3), i//3) for i in range(5*2*3)]
    query_labels = [(i//(15*3), i//3) for i in range(15*5*3)]
    acc = ncm(support_features, query_features, support_labels, query_labels)
    print(acc)

if __name__ == '__main__':
    test()
