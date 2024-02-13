import torch
import torch.nn as nn


import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if path not in sys.path:sys.path.append(path)


class MLP(torch.nn.Module):
    def __init__(self, support_features,unique_labels,logit_scale=4.60517):
        super(MLP, self).__init__()
        self.head = nn.Linear(support_features.shape[1], len(unique_labels))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logit_scale = torch.FloatTensor([logit_scale]).to(device)
    def forward(self, x):
        x = self.head(x)
        return x * self.logit_scale.exp()