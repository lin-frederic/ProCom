import torch
import torch.nn.functional as F

class preprocess_NCM(torch.nn.Module):
    # preprocess_NCM: preprocess the features for NCM
    # including: 1. subtract the mean feature vector
    #            2. normalize the feature vector to make it into hypersphere
    # input: support_features: {class1: {features: [n_shot, d], indices: [n_shot]}, ...}
    #        query_features: {class1: {features: [n_query, d], indices: [n_query]}, ...}
    # output: support_output: [(class,index (shot),feature), ...]
    #         sphered_query_features: [(class,index (query),feature), ...]
    def __init__(self):
        super(preprocess_NCM, self).__init__()
    def forward(self, support_features, query_features):
        all_support_features = []
        for class_id in support_features:
            all_support_features.extend(support_features[class_id]["features"]) # [n_way * n_shot, d], n_shot may be different for each class
        mean_feature = torch.mean(torch.stack(all_support_features), dim=0) # [d]
        sphered_support_features = {} # {class: [n_shot, d], ...}
        sphered_query_features = {} # {class: [n_query, d], ...}
        for class_label, class_features in support_features.items():
            sphered_support_features[class_label] = class_features["features"] - mean_feature
            sphered_support_features[class_label] = F.normalize(sphered_support_features[class_label], p=2, dim=1) # [n_shot, d]
            sphered_support_features[class_label] = {"features": sphered_support_features[class_label], "indices": class_features["indices"]}
        for class_label, class_features in query_features.items():
            sphered_query_features[class_label] = class_features["features"] - mean_feature
            sphered_query_features[class_label] = F.normalize(sphered_query_features[class_label], p=2, dim=1) # [n_query, d]
            sphered_query_features[class_label] = {"features": sphered_query_features[class_label], "indices": class_features["indices"]}
        support_output = [] # [(class,index,feature), ...]
        query_output = [] # [(class,index,feature), ...]
        for class_label, class_features in sphered_support_features.items():
            for index in class_features["indices"]:
                support_output.append((class_label, index, class_features["features"][index]))
        for class_label, class_features in sphered_query_features.items():
            for index in class_features["indices"]:
                query_output.append((class_label, index, class_features["features"][index]))
        return support_output, query_output
class NCM(torch.nn.Module):
    def __init__(self):
        super(NCM, self).__init__()

    def forward(self, support_features, query_features, use_cosine=True):
        # support_features: [(class,shot,feature), ...)]
        # query_features: [(class,query,feature), ...)]
        # output: [n_query,class] each element is the predicted class for each query

        # Preprocess the features
        preprocess_NCM_layer = preprocess_NCM()
        support_features, query_features = preprocess_NCM_layer(support_features, query_features)
        acc = 0
        for query_class, query_index, query_feature in query_features:
            max_sim = float("-inf")
            max_class = None
            for support_class, support_index, support_feature in support_features:
                if use_cosine:
                    sim = torch.dot(query_feature, support_feature)
                else:
                    sim = -torch.norm(query_feature - support_feature, p=2)
                if sim > max_sim:
                    max_sim = sim
                    max_class = support_class
            if max_class == query_class:
                acc += 1
        acc = acc / len(query_features)
        return acc
        



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
