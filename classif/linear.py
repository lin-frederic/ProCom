import torch
from tools import preprocess_Features
from tqdm import tqdm
class Linear(torch.nn.Module):
    def __init__(self, top_k=1,n_epochs=40000,lr = 0.001, temperature = 0.01):
        super(Linear, self).__init__()
        self.preprocess_layer = preprocess_Features()
        self.top_k = top_k
        self.n_epochs = n_epochs
        self.lr = lr
        self.temperature = temperature
    def forward(self, support_features, query_features, support_labels, query_labels, temp_query_labels, encode_labels=False):
        # support_features: list of features, as a tensor of shape [n_shot, d]
        # support_labels: list of (class,image_index)
        #support_features, query_features = self.preprocess_layer(support_features, query_features)
        # experiments shows that the preprocess layer makes the performance worse
        if encode_labels:
            train_labels, support_annotation_idx = [label[0] for label in support_labels], [label[1] for label in support_labels]
            query_labels, query_annotation_idx = [label[0] for label in query_labels], [label[1] for label in query_labels]
            unique_labels = list(set(train_labels))
            train_labels = [unique_labels.index(label) for label in train_labels]
            query_labels = [unique_labels.index(label) for label in query_labels]
            original_query_labels = [unique_labels.index(label) for label in temp_query_labels]
            train_labels = torch.tensor(train_labels)
            query_labels = torch.tensor(query_labels)
        classifier = torch.nn.Linear(support_features.shape[1], len(unique_labels))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=self.lr)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classifier.to(device)
    
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            output = classifier(support_features.to(device))
            loss = criterion(output, torch.tensor(train_labels).to(device))
            loss.backward()
            optimizer.step()
        outputs = classifier(support_features.to(device))
        # transform into logit
        outputs = torch.nn.functional.softmax(outputs/self.temperature, dim=1)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == train_labels.to(device)).sum().item() / len(train_labels)
        print(f"Accuracy on support set: {acc}")
        acc = 0
        outputs = classifier(query_features.to(device))
        # transform into logit
        outputs = torch.nn.functional.softmax(outputs/self.temperature, dim=1)
        img_classif = {}
        for i in range(len(query_labels)):
            img_idx = query_annotation_idx[i]
            if img_idx not in img_classif:
                img_classif[img_idx] = []
            # find the index of the highest logit for this annotation
            class_idx = torch.argmax(outputs[i]).item()
            img_classif[img_idx].append((class_idx, outputs[i][class_idx].item())) # class_idx, score
        for img in img_classif:
                # find the class with the highest score
                idx = 0
                for i in range(len(img_classif[img])):
                    if img_classif[img][i][1] > img_classif[img][idx][1]:
                        idx = i
                img_classif[img] = img_classif[img][idx][0]
                acc += int(img_classif[img] == original_query_labels[img])
        acc = acc/len(img_classif)
        return acc