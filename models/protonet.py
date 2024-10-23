import torch
import torch.nn as nn
import torch.nn.functional as F

class ProtoNet(nn.Module):
    def __init__(self, backbone, emb_dim):
        super(ProtoNet, self).__init__()
        self.embedding = backbone
        self.emb_dim = emb_dim
        self.temperature = 1/64
        self.temp_for_constrative = 1/32
        
    def forward(self, x_support, x_query, num_ways):
        support_set = self.embedding(x_support)
        query_set = self.embedding(x_query)
        
        #for prediction
        prototypes = torch.mean(support_set.view(num_ways, -1, self.emb_dim), dim=1).squeeze(1)
        distances = torch.cdist(query_set, prototypes)*self.temperature
        cosine_sim = torch.einsum('qd, nd -> qn', query_set, prototypes)*self.temp_for_constrative
        
        return -distances
