import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import random
from sklearn.cluster import KMeans

def make_cactus_paritions(trainloader, n_clusters, device):
    encoder = models.resnet18(pretrained=True)
    encoder.fc = nn.Linear(512, 256)
    encoder = encoder.to(device)
    embedding_list = []
    
    for data in trainloader:
        inputs, labels = data
        inputs = inputs.to(device)
        embedding_list.append(encoder(inputs).detach())
    
    embeddings = torch.concat(embedding_list, dim=0)
    
    partitions = KMeans(n_clusters=n_clusters, n_init=10, max_iter=3000, init='k-means++').fit(embeddings.cpu()).labels_
    labels = torch.LongTensor(partitions)
    
    return labels

