import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import random


class CACTUS():
    def __init__(self, trainloader, num_partitions, device):
        super(CACTUS, self).__init__()
        self.device = device
        self.embeddings, self.images = self.encoding_images(trainloader)
        self.partitions = self.make_partitions(num_partitions, self.embeddings)
        
    def encoding_images(self, trainloader):
        encoder = models.resnet18(pretrained=True)
        encoder.fc = nn.Linear(512, 256)
        encoder = encoder.to(self.device)
        embedding_list = []
        images = []
        
        for data in trainloader:
            inputs, labels = data
            inputs = inputs.to(self.device)
            embedding_list.append(encoder(inputs).detach())
            images.append(inputs)
        
        embeddings = torch.concat(embedding_list, dim=0)
        images = torch.concat(images, dim=0)
        
        return embeddings, images
    
    def Kmeans(self, k, embeddings, num_init, max_iter):
        min_variance = float('inf')
        opt_cluster_indices = None
        
        for _ in range(num_init):
            mu = torch.stack(random.sample(list(embeddings), k))
            
            for _ in range(max_iter):
                distances = torch.cdist(embeddings, mu)
                cluster_indices = torch.argmin(distances, dim=1)
                
                new_mu = []
                for idx in range(k):
                    cluster_points = embeddings[cluster_indices == idx]
                    if len(cluster_points)>0:
                        new_mu.append(cluster_points.mean(dim=0))
                    else:
                        new_mu.append(mu[idx])
                new_mu = torch.stack(new_mu)
                
                if torch.allclose(mu, new_mu, atol=1e-4):
                    break
                mu = new_mu
            
            init_variance = torch.sum(torch.min(torch.cdist(embeddings, mu), dim=1).values)
            
            if init_variance < min_variance:
                min_variance = init_variance
                opt_cluster_indices = cluster_indices
        
        return opt_cluster_indices
    
    def make_partitions(self, num_partitions, embeddings):
        partitions = []
        for i in range(num_partitions):
            partitions.append(self.Kmeans(500, embeddings, num_init=10, max_iter=3000))
        return partitions
    
    def create_task_kmeans(self, num_tasks, num_ways, num_shots, num_queries):
        tasks =[]
        for i in range(num_tasks):
            partition_idx = torch.randperm(len(self.partitions))[0]
            kmeans = self.partitions[partition_idx]
            
            clusters = [[] for i in range(500)]
            
            for img, label in zip(self.images, kmeans):
                clusters[label].append(img)
            clusters = [cluster for cluster in clusters if len(cluster)>=(num_shots+num_queries)]
            
            batch_support_set = torch.zeros(num_ways * num_shots, self.images.size(1), self.images.size(2), self.images.size(3)).to(self.device)
            batch_query_set = torch.zeros(num_ways * num_queries, self.images.size(1), self.images.size(2), self.images.size(3)).to(self.device)
            class_indices = torch.randperm(len(clusters))[:num_ways]
            
            for i, label in enumerate(class_indices):
                slice_for_support = slice(i*num_shots, (i+1)*num_shots)
                slice_for_queries = slice(i*num_queries, (i+1)*num_queries)
                
                samples = random.sample(clusters[label], num_shots+num_queries)
                batch_support_set[slice_for_support] = torch.stack(samples[:num_shots])
                batch_query_set[slice_for_queries] = torch.stack(samples[num_shots:])
            
            y_support_set = torch.arange(0, num_ways, 1 / num_shots).long().to(self.device)
            y_query_set = torch.arange(0, num_ways, 1 / num_queries).long().to(self.device)
            
            task = [batch_support_set, batch_query_set, y_support_set, y_query_set]
            tasks.append(task)
            
        return tasks



