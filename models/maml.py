import torch
import torch.nn as nn
import torch.nn.functional as F
import collections

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

def make_convblock(input, w_conv, b_conv, w_bn, b_bn):
    x = F.conv2d(input, w_conv, b_conv, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)
    x = F.relu(x)
    output = F.max_pool2d(x, kernel_size=2, stride=2)
    
    return output

class CNNclasifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNclasifier, self).__init__()
        self.num_classes = num_classes
        self.layer1 = ConvBlock(3, 32)
        self.layer2 = ConvBlock(32, 32)
        self.layer3 = ConvBlock(32, 32)
        self.layer4 = ConvBlock(32, 32)
        self.fc = nn.Linear(32*5*5, num_classes)

    def forward(self,x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        
        return y
    
    def params_forward(self, x, params):
        for block in range(1,5):
            x = make_convblock(x, params[f"layer{block}.conv2d.weight"], params[f"layer{block}.conv2d.bias"], 
                            params.get(f"layer{block}.bn.weight"), params.get(f"layer{block}.bn.bias"))
        x = x.view(x.size(0), -1)
        x = F.linear(x, params["fc.weight"], params["fc.bias"])
        return x

class MAML(nn.Module):
    def __init__(self, inner_lr, num_ways):
        super(MAML, self).__init__()
        self.classifier = CNNclasifier(num_ways)
        self.inner_lr = inner_lr
    
    def forward(self, tasks, num_inner_steps):
        total_loss = 0
        
        for x_support, x_query, y_support, y_query in tasks:
            fast_weights = collections.OrderedDict(self.classifier.named_parameters())
            
            for i in range(num_inner_steps):
                logits_support = self.classifier.params_forward(x_support, fast_weights)
                loss_support = F.cross_entropy(logits_support, y_support)
                grads = torch.autograd.grad(loss_support, fast_weights.values(), create_graph=True)
                fast_weights = collections.OrderedDict((name, param - self.inner_lr * grads) for ((name, param), grads) in zip(fast_weights.items(), grads))
            
            logits_query = self.classifier.params_forward(x_query, fast_weights)
            loss_query = F.cross_entropy(logits_query, y_query)
            total_loss += loss_query
        
        return total_loss/len(tasks), logits_query