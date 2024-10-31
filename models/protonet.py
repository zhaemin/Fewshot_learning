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
        
    def forward(self, tasks, num_ways):
        x_support, x_query, y_support, y_query = tasks[0]
        support_set = self.embedding(x_support)
        query_set = self.embedding(x_query)
        
        #for prediction
        prototypes = torch.mean(support_set.view(num_ways, -1, self.emb_dim), dim=1)
        logits = -torch.cdist(query_set, prototypes)*self.temperature
        #prototypes = F.normalize(prototypes, dim=-1)
        #cosine_sim = torch.einsum('qd, nd -> qn', query_set, prototypes)*self.temp_for_constrative
        
        loss = F.cross_entropy(logits, y_query)
        
        return loss, logits
