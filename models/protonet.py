import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import split_support_query_set

class ProtoNet(nn.Module):
    def __init__(self, backbone, emb_dim):
        super(ProtoNet, self).__init__()
        self.embedding = backbone
        self.emb_dim = emb_dim
        
    def forward(self, args, inputs, labels, num_ways, device):
        inputs = self.embedding(inputs)
        
        if args.unsupervised == 'umtra' and self.training:
            tasks = split_support_query_set(inputs, labels, num_ways, 1, 1, args.num_tasks, self.training, device)
        else:
            tasks = split_support_query_set(inputs, labels, num_ways, args.num_shots, args.num_queries, args.num_tasks, self.training, device)
            
        total_loss = 0
        
        for task in tasks:
            support_set, query_set, y_support, y_query = task
            
            #for prediction
            prototypes = torch.mean(support_set.view(num_ways, -1, self.emb_dim), dim=1)
            logits = -torch.cdist(query_set, prototypes)
            
            loss = F.cross_entropy(logits, y_query)
            total_loss += loss
            
        acc = None
        if not self.training:
            with torch.no_grad():
                _, predicted = torch.max(logits.data,1)
                total = args.num_queries * args.test_num_ways
                correct = (predicted == y_query).sum().item()
                acc = 100*correct/total
                
        return total_loss/len(tasks), acc
