import torch
import torch.nn as nn
import torch.nn.functional as F

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
        att = torch.softmax(torch.einsum('qd, kd -> qk', q, k)*self.scaling, dim=-1)
        y = torch.einsum('qk, vd -> qd', att, v)
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
        
    def forward(self, x_support, x_query, num_ways):
        support_set = self.embedding(x_support)
        query_set = self.embedding(x_query)
        
        #for prediction
        prototypes = torch.mean(support_set.view(num_ways, -1, self.emb_dim), dim=1).squeeze(1) # 5 shots 640 -> 5 640
        prototypes_transform = self.attention(prototypes) 
        distances = torch.cdist(query_set, prototypes_transform)*self.temp
        
        #for contrastive loss
        if self.training:
            support_set = support_set.view(num_ways, -1, self.emb_dim)
            query_set = query_set.view(num_ways, -1, self.emb_dim)
            aux_set = torch.cat((support_set, query_set), dim=1).view(-1, self.emb_dim) #num_ways * num_support_set+num_query_set * 640
            
            aux_set_transform = self.attention(aux_set).view(num_ways, -1, self.emb_dim)
            aux_center = torch.mean(aux_set_transform, dim=1)
            aux_set_transform = aux_set_transform.view(-1, self.emb_dim)
            aux_distances = torch.cdist(aux_set_transform, aux_center)*self.temp_for_constrative
            #aux_set_transform = F.normalize(aux_center, dim=-1)
            #cosine_sim = torch.einsum('qd, nd -> qn', aux_set_transform, aux_center)*self.temp_for_constrative
            
            return -distances, -aux_distances
        else:
            return -distances