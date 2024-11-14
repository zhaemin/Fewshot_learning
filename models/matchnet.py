import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import split_support_query_set

class MatchNet(nn.Module):
    def __init__(self, backbone):
        super(MatchNet, self).__init__()
        self.embedding = backbone
        self.temperature = 1
        
    def forward(self, args, inputs, labels, num_ways, device):
        inputs = self.embedding(inputs)
        tasks = split_support_query_set(inputs, labels, num_ways, args.num_shots, args.num_queries, args.num_tasks, self.training, device)
        total_loss = 0
        
        for task in tasks:
            support_set, query_set, y_support, y_query = task
            #for prediction
            one_hot_label = F.one_hot(y_support).float() # num_support_set * num_ways
            attention = torch.softmax(F.cosine_similarity(support_set.unsqueeze(0), query_set.unsqueeze(1), dim=-1), dim=1) # num_query_set * num_support_set
            logits = torch.einsum('qs, sn -> qn', attention, one_hot_label)
            
            loss = F.cross_entropy(logits, y_query)
            total_loss += loss
        
        with torch.no_grad():
            _, predicted = torch.max(logits.data,1)
            total = args.num_queries * args.test_num_ways
            correct = (predicted == y_query).sum().item()
                
        return total_loss/len(tasks), 100*correct/total
