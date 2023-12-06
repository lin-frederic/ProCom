
import torch
import torch.nn.functional as F
from tools import preprocess_Features


class KMeans(torch.nn.Module):
    def __init__(self):
        super(KMeans, self).__init__()
        self.preprocess_KMeans_layer = preprocess_Features()

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


    def forward(self, support_features, query_features, support_labels, query_labels, n_iter=50, beta=5):
        # support_features: list of features, as a tensor of shape [n_shot, d]
        # support_labels: list of (class,image_index) tuples
        # query_features: list of features, as a tensor of shape [n_query, d]
        # output: [n_query,class] each element is the predicted class for each query
        
        # Preprocess the features
        support_output, query_output = self.preprocess(support_features, query_features)
        acc = 0

        # initialize centroids
        centroids = {}
        for class_label, _, class_features in support_output:
            centroids[class_label] = torch.mean(class_features, dim=0)
        
        # update centroids
        for _ in range(n_iter):
            new_centroids = {} 
            for centroid in centroids:
                suma = 0
                sumb = 0
                for class_label, _ , class_feature in support_output:
                    if class_label == centroid:
                        suma += class_feature
                        sumb += 1
                for class_label, _ , class_feature in query_output:
                    weight = self.w(class_feature, centroids, centroid, beta)
                    suma += weight * class_feature
                    sumb += weight
                new_centroids[centroid] = suma / sumb
            centroids = new_centroids

        # assign query features to centroids with least distance
        for class_label, _ , class_feature in query_output:
            min_distance = float("inf")
            min_centroid = None
            for centroid in centroids:
                distance = torch.norm(class_feature - centroids[centroid])
                if distance < min_distance:
                    min_distance = distance
                    min_centroid = centroid
            if min_centroid == class_label:
                acc += 1
        acc /= len(query_output)
        return acc
                        



def test():
    # Test KMeans
    kmeans = KMeans()
    support_features = torch.randn(5*2*3, 10)
    query_features = torch.randn(15*5*3, 10)
    support_labels = [(i//(2*3), i//3) for i in range(5*2*3)]
    query_labels = [(i//(15*3), i//3) for i in range(15*5*3)]
    acc = kmeans(support_features, query_features, support_labels, query_labels)
    print(acc)

if __name__ == "__main__":
    test()
   



