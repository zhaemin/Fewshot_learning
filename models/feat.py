import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import split_support_query_set

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention,self).__init__()
        self.make_qkv = nn.Linear(dim, 3*dim, bias=False)
        self.scaling = dim**(-0.5)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
        nn.init.xavier_normal_(self.make_qkv.weight)
        nn.init.xavier_normal_(self.fc.weight)
        
    def forward(self, x):
        qkv = self.make_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1) 
        att = torch.softmax(torch.einsum('bqd, bkd -> bqk', q, k)*self.scaling, dim=-1)
        y = torch.einsum('bqk, bvd -> bqd', att, v)
        y = self.dropout(self.fc(y))
        y = self.norm(x+y)
        return y

class FEAT(nn.Module):
    def __init__(self, backbone, emb_dim):
        super(FEAT, self).__init__()
        self.embedding = backbone
        self.emb_dim = emb_dim
        self.attention = SelfAttention(emb_dim)
        self.temp = 1/64
        self.temp_for_constrative = 1/32
        
    def forward(self, args, inputs, labels, num_ways, device):
        inputs = self.embedding(inputs)
        tasks = split_support_query_set(inputs, labels, num_ways, args.num_shots, args.num_queries, args.num_tasks, self.training, device)
        total_loss = 0
        
        for task in tasks:
            support_set, query_set, y_support, y_query = task
        
            #for prediction
            prototypes = torch.mean(support_set.view(num_ways, -1, self.emb_dim), dim=1).unsqueeze(0)
            prototypes_transform = self.attention(prototypes).view(-1, self.emb_dim)
            logits = -torch.cdist(query_set, prototypes_transform)*self.temp
            
            loss = F.cross_entropy(logits, y_query)
            
            #for contrastive loss
            if self.training:
                support_set = support_set.view(num_ways, -1, self.emb_dim)
                query_set = query_set.view(num_ways, -1, self.emb_dim)
                aux_set = torch.cat((support_set, query_set), dim=1).view(num_ways, -1, self.emb_dim) #num_ways * num_support_set+num_query_set * 640
                aux_set_transform = self.attention(aux_set)
                
                aux_center = torch.mean(aux_set_transform, dim=1)
                aux_set_transform = aux_set_transform.view(-1, self.emb_dim)
                aux_logits = -torch.cdist(aux_set_transform, aux_center)*self.temp_for_constrative
                aux_y = torch.cat((y_support.view(num_ways, -1),y_query.view(num_ways, -1)),dim=-1).view(-1)
                #aux_set_transform = F.normalize(aux_center, dim=-1)
                #cosine_sim = torch.einsum('qd, nd -> qn', aux_set_transform, aux_center)*self.temp_for_constrative
                
                loss += F.cross_entropy(aux_logits, aux_y)*0.1
                total_loss += loss
            
        with torch.no_grad():
            _, predicted = torch.max(logits.data,1)
            total = args.num_queries * args.test_num_ways
            correct = (predicted == y_query).sum().item()
                
        return total_loss/len(tasks), 100*correct/total