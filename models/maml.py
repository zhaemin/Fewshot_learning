import torch
import torch.nn as nn
import torch.nn.functional as F
import collections

from utils import split_support_query_set

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x

class CNNclasifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNclasifier, self).__init__()
        self.num_classes = num_classes
        self.layer1 = ConvBlock(3, 32)
        self.layer2 = ConvBlock(32, 32)
        self.layer3 = ConvBlock(32, 32)
        self.layer4 = ConvBlock(32, 32)
        self.fc = nn.Linear(32*5*5, num_classes)
        
        self.register_buffer('running_mean', torch.zeros(4, 32, requires_grad=False))
        self.register_buffer('running_var', torch.ones(4, 32, requires_grad=False))

    def forward(self,x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        
        return y
    
    def reset_buffers(self):
        self.running_mean = torch.zeros_like(self.running_mean)
        self.running_var = torch.ones_like(self.running_var)
        
    # inner loop에서 학습할 때 running_mean과 running_var 갱신하고, outer loop에서 query set으로 initial param 업데이트할 때 해당 값을 사용
    def make_convblock(self, input, w_conv, b_conv, w_bn, b_bn, running_mean, running_var, training):
        x = F.conv2d(input, w_conv, b_conv, padding=1)
        x = F.batch_norm(x, weight=w_bn, bias=b_bn, running_mean=running_mean, running_var=running_var, training=training)
        x = F.relu(x)
        output = F.max_pool2d(x, kernel_size=2, stride=2)
        return output
    
    def params_forward(self, x, params, training):
        for block in range(1,5):
            x = self.make_convblock(x, params.get(f"layer{block}.conv2d.weight"),
                                    params.get(f"layer{block}.conv2d.bias"), 
                                    params.get(f"layer{block}.bn.weight"), 
                                    params.get(f"layer{block}.bn.bias"),
                                    self.running_mean[block-1],
                                    self.running_var[block-1],
                                    training)
        x = x.view(x.size(0), -1)
        x = F.linear(x, params["fc.weight"], params["fc.bias"])
        return x

class MAML(nn.Module):
    def __init__(self, inner_lr, num_ways):
        super(MAML, self).__init__()
        self.classifier = CNNclasifier(num_ways)
        self.inner_lr = inner_lr
        
    def forward(self, args, inputs, labels, num_ways, device):
        
        if args.unsupervised == 'umtra':
            tasks = split_support_query_set(inputs, labels, num_ways, 1, 1, args.num_tasks, self.training, device)
        else:
            tasks = split_support_query_set(inputs, labels, num_ways, args.num_shots, args.num_queries, args.num_tasks, self.training, device)
        
        if self.training:
            num_inner_steps = 5
        else:
            num_inner_steps = 50
            
        total_loss = 0
        
        for x_support, x_query, y_support, y_query in tasks:
            fast_weights = collections.OrderedDict(self.classifier.named_parameters())
            self.classifier.reset_buffers()
            
            for i in range(num_inner_steps):
                logits_support = self.classifier.params_forward(x_support, fast_weights, True)
                loss_support = F.cross_entropy(logits_support, y_support)
                grads = torch.autograd.grad(loss_support, fast_weights.values(), create_graph=True)
                fast_weights = collections.OrderedDict((name, param - self.inner_lr * grads) for ((name, param), grads) in zip(fast_weights.items(), grads))
            
            logits_query = self.classifier.params_forward(x_query, fast_weights, False)
            loss_query = F.cross_entropy(logits_query, y_query)
            total_loss += loss_query
        
        acc = None
        if not self.training:
            with torch.no_grad():
                _, predicted = torch.max(logits_query.data,1)
                total = args.num_queries * args.test_num_ways
                correct = (predicted == y_query).sum().item()
                acc = 100*correct/total
                
        return total_loss/len(tasks), acc