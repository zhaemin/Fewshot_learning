import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchNet(nn.Module):
    def __init__(self, backbone):
        super(MatchNet, self).__init__()
        self.embedding = backbone
        self.temperature = 1
        
    def forward(self, tasks, num_ways):
        x_support, x_query, y_support, y_query = tasks[0]
        support_set = self.embedding(x_support)
        query_set = self.embedding(x_query)
        
        #for prediction
        one_hot_label = F.one_hot(y_support).float() # num_support_set * num_ways
        attention = torch.softmax(F.cosine_similarity(support_set.unsqueeze(0), query_set.unsqueeze(1), dim=-1), dim=1) # num_query_set * num_support_set
        logits = torch.einsum('qs, sn -> qn', attention, one_hot_label)
        
        loss = F.cross_entropy(logits, y_query)
        
        return loss, logits
