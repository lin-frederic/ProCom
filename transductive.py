
import torch
import torch.nn.functional as F
import time

class preprocess_KMeans(torch.nn.Module):
    # preprocess_KMeans: preprocess the features for KMeans
    # including: 1. subtract the mean feature vector
    #            2. normalize the feature vector to make it into hypersphere
    # input: support_features: {class1: {features: [n_shot, d], indices: [n_shot]}, ...}
    #        query_features: {class1: {features: [n_query, d], indices: [n_query]}, ...}
    # output: support_output: [(class,index (shot),feature), ...]
    #         sphered_query_features: [(class,index (query),feature), ...]
    def __init__(self):
        super(preprocess_KMeans, self).__init__()
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
    


class KMeans(torch.nn.Module):
    def __init__(self):
        super(KMeans, self).__init__()

    def w(self, features, centroids, centroid_i, beta):
        # features: [n, d]
        # centroids: [n_clusters, d]
        # beta: scalar
        # return: [n, n_clusters]

        # softmax
        # initialize distances
        distances = {}
        for centroid in centroids:
            distances[centroid] = torch.norm(features - centroids[centroid])

        sum_distances = torch.zeros_like(distances[centroid_i])
        for centroid in centroids:
            sum_distances += torch.exp(-beta * distances[centroid] ** 2)
        return torch.exp(-beta * distances[centroid_i] ** 2) / sum_distances


    def forward(self, support_features, query_features, n_iter, beta):

        # Preprocess the features
        preprocess = preprocess_KMeans()
        support_output, query_output = preprocess(support_features, query_features)
        acc = 0

        # initialize centroids
        centroids = {}
        for class_label, class_features in support_features.items():
            centroids[class_label] = torch.mean(class_features["features"], dim=0)
        
        # update centroids
        for i in range(n_iter):
            for centroid in centroids:
                suma = 0
                sumb = 0
                for class_label, _ , class_feature in support_output:
                    if class_label == centroid:
                        suma += 1
                        sumb += 1
                for class_label, _ , class_feature in query_output:
                    if class_label == centroid:
                        weight = self.w(class_feature, centroids, centroid, beta)
                        suma += weight * class_feature
                        sumb += weight
                centroids[centroid] = suma / sumb

        # predict
        for query_class, query_index, query_feature in query_output:
            max_sim = float("-inf")
            max_class = None
            for support_class, support_index, support_feature in support_output:
                sim = torch.dot(query_feature, support_feature)
                if sim > max_sim:
                    max_sim = sim
                    max_class = support_class
            if max_class == query_class:
                acc += 1

        acc = acc / len(query_output)
        return acc
                        



def test():
    kmeans = KMeans()
    support_features = {}
    query_features = {}
    for i in range(5):
        support_features[i] = {"features": torch.randn(5, 10), "indices": [0, 1, 2, 3, 4]}
        query_features[i] = {"features": torch.randn(5, 10), "indices": [0, 1, 2, 3, 4]}
    print(kmeans(support_features, query_features, 10, 2))

if __name__ == "__main__":
    test()
    

