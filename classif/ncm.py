import torch
import torch.nn.functional as F
import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if path not in sys.path:sys.path.append(path)
from tools import preprocess_Features
import matplotlib.pyplot as plt

class NCM(torch.nn.Module):
    def __init__(self, top_k=1):
        super(NCM, self).__init__()
        self.preprocess_NCM_layer = preprocess_Features()
        self.top_k = top_k

    def forward(self, support_features, query_features, support_labels, query_labels, use_cosine=False,to_display=None):
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
        if to_display is not None:
            support_augmented_imgs, query_augmented_imgs = to_display          
        for i, (query_class, query_index) in enumerate(query_labels):
            #find where the other masks of the same image are
            if query_index not in visited:
                visited.add(query_index)
                indices = [j for j, (class_, index_) in enumerate(query_labels) if index_ == query_index]
                query_similarity = similarity[indices, :] # [n_masks, n_shot]
                n_masks, n_shot = query_similarity.shape 
                query_similarity = query_similarity.flatten()
                # the first n_shot elements are for the first mask, the next n_shot for the second mask, etc
                sorted_similarity, sorted_indices = torch.sort(query_similarity, descending=True)
                # get the top k indices
                top_k_indices = sorted_indices[:self.top_k]
                # get the top k masks, and the top k corresponding support features
                top_k_masks = top_k_indices // n_shot
                top_k_support = top_k_indices % n_shot
                if to_display is not None:
                    #plot the query image and the support image (top 1)
                    #plot the query mask and the support mask (top 1)
                    # set title as correct or not
                    fig, axs = plt.subplots(1,2)
                    query_img = query_augmented_imgs[indices[top_k_masks[0]]]
                    support_img = support_augmented_imgs[top_k_support[0]]
                    query_img = (query_img - query_img.min()) / (query_img.max() - query_img.min())
                    support_img = (support_img - support_img.min()) / (support_img.max() - support_img.min())
                    query_img = (query_img * 255).astype('uint8')
                    support_img = (support_img * 255).astype('uint8')
                    axs[0].imshow(query_img)
                    axs[0].axis('off')
                    axs[1].imshow(support_img)
                    axs[1].axis('off')
                    if query_class == support_labels[top_k_support[0]][0]:
                        fig.suptitle("Correct")
                    else:
                        fig.suptitle("Incorrect")
                    plt.savefig(f"results/ncm/{query_index}.png")
                    plt.close()
                    
                top_k_classes = [support_labels[i][0] for i in top_k_support]
                # calculate top k accuracy
                if query_class in top_k_classes:
                    acc += 1
        acc = acc / len(visited)
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
if __name__ == '__main__':
    test2()
