import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MoCo(nn.Module):
    def __init__(self, q_size, momentum):
        super(MoCo, self).__init__()
        self.dim = 128
        self.f_q = self.make_encoder()
        self.f_k = self.make_encoder()
        self.queue = None
        self.init_flag = True
        self.max_queue_size = q_size
        self.momentum = momentum
        
        for k_param, q_param in zip(self.f_k.parameters(), self.f_q.parameters()):
            k_param.data.copy_(q_param.data)
            k_param.requires_grad = False
        
    def make_encoder(self):
        f = models.resnet50()
        f.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        f.maxpool = nn.Identity()
        f.fc = nn.Linear(2048, self.dim)
        
        return f
    
    def forward(self, args, inputs, labels, num_ways, device):
        #pretext task
        if args.pretext:
            batch_size = inputs.size(0)
            
            inputs = inputs.transpose(0, 1)
            x_q = inputs[0]
            x_k = inputs[1]
            
            if not self.init_flag:
                with torch.no_grad():
                    for k_param, q_param in zip(self.f_k.parameters(), self.f_q.parameters()):
                        k_param.data = self.momentum * k_param.data + (1 - self.momentum) * q_param.data
            
            q = F.normalize(self.f_q(x_q), dim=1)
            k = F.normalize(self.f_k(x_k), dim=1)
            k = k.detach()
            
            loss = None
            if self.queue is not None:
                logits_positive = torch.bmm(q.view(batch_size, 1, self.dim), k.view(batch_size, self.dim, 1)).squeeze(-1)
                logits_negative = torch.mm(q.view(batch_size, self.dim), self.queue.view(self.dim, -1))
                
                logits = torch.cat((logits_positive, logits_negative), dim=1)
                
                sim_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
                loss = F.cross_entropy(logits/0.07, sim_labels)
            
            if self.init_flag:
                self.queue = k
                self.init_flag = False
            else:
                self.queue = torch.cat((self.queue, k), dim=0)
            
            if self.queue.size(0) > self.max_queue_size:
                self.queue = self.queue[batch_size:]
            
            acc = 0
        
        #classification task
        else:
            for param in self.f_q.parameters():
                param.requires_grad = False
            
            self.f_q.fc = nn.Linear(2048, num_ways).to(device)
            logits = self.f_q(inputs)
            
            loss = F.cross_entropy(logits, labels)
            
            acc = None
            if not self.training:
                with torch.no_grad():
                    _, predicted = torch.max(logits.data, 1)
                    total = labels.size(0)
                    correct = (predicted == labels).sum().item()
                    acc = 100 * correct / total
                    
        return loss, acc

