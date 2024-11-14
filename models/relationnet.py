import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import split_support_query_set

class ConvBlock(nn.Sequential):
    def __init__(self, in_features, out_features, padding):
        super(ConvBlock, self).__init__(
            nn.Conv2d(in_features, out_features, kernel_size=3, padding=padding),
            nn.BatchNorm2d(out_features),
            nn.ReLU()
            )

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = ConvBlock(3, 64, 0)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.layer2 = ConvBlock(64, 64, 0)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.layer3 = ConvBlock(64, 64, 1)
        self.layer4 = ConvBlock(64, 64, 1)
        
    def forward(self,x):
        y = self.layer1(x)
        y = self.pool1(y)
        y = self.layer2(y)
        y = self.pool2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        return y # 64

class RelationBlock(nn.Module):
    def __init__(self):
        super(RelationBlock, self).__init__()
        self.conv1 = ConvBlock(128, 64, 0)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = ConvBlock(64, 64, 0)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.FC1 = nn.Linear(64*3*3, 8)
        self.FC2 = nn.Linear(8, 1)
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.pool2(y)
        y = y.view(y.size(0), -1)
        y = F.relu(self.FC1(y))
        y = F.sigmoid(self.FC2(y))
        
        return y

class RelationNet(nn.Module):
    def __init__(self):
        super(RelationNet, self).__init__()
        self.encoder = CNNEncoder() # n * 64 * 19 * 19
        self.relationblock = RelationBlock()
        
    def forward(self, args, inputs, labels, num_ways, device):
        inputs = self.encoder(inputs)
        tasks = split_support_query_set(inputs, labels, num_ways, args.num_shots, args.num_queries, args.num_tasks, self.training, device)
        total_loss = 0
        
        for task in tasks:
            support_set, query_set, y_support, y_query = task
            
            support_set = torch.sum(support_set.view(num_ways, -1, 64, 19, 19), dim=1) # num_ways 64 19 19
            support_set = support_set.unsqueeze(0).repeat(args.num_queries*num_ways, 1, 1, 1, 1) # num_support_set num_ways 64 19 19
            query_set = torch.transpose(query_set.unsqueeze(0).repeat(num_ways, 1, 1, 1, 1), 0, 1) # num_support_set num_ways 64 19 19
            
            relation_input = torch.cat((support_set, query_set), dim=2).view(-1, 128, 19, 19) # num_support_set*num_queries 128 19 19
            relation_output = self.relationblock(relation_input).view(-1, num_ways)
            
            one_hot_labels = F.one_hot(y_query).float()
            loss = F.mse_loss(relation_output, one_hot_labels)
            total_loss += loss
        
        with torch.no_grad():
            _, predicted = torch.max(relation_output.data,1)
            total = args.num_queries * args.test_num_ways
            correct = (predicted == y_query).sum().item()
                
        return total_loss/len(tasks), 100*correct/total