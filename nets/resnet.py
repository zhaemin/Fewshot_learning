import torch
import torch.nn as nn
import torch.nn.functional as F

class PreactResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(PreactResidualBlock,self).__init__()
        
        if in_features == out_features:
            stride = 1
            self.identity = nn.Identity()
        else:
            stride = 2
            self.identity = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=2, bias=False),
                nn.BatchNorm2d(out_features)
                )
        
        self.batchnorm1 = nn.BatchNorm2d(in_features)
        self.conv1 = nn.Conv2d(in_features, out_features, 3, stride=stride, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_features)
        self.conv2 = nn.Conv2d(out_features, out_features, 3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        y = self.batchnorm1(x)
        y = self.relu(y)
        y = self.conv1(y)
        y = self.batchnorm2(y)
        y = self.relu(y)
        y = self.conv2(y)
        y += self.identity(x)
        
        return y

class PreactBottleneckBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(PreactBottleneckBlock,self).__init__()
        
        if in_features == out_features*4:
            stride = 1
            self.identity = nn.Identity()
        elif in_features == out_features:
            stride = 1
            self.identity = nn.Sequential(
                nn.Conv2d(in_features, out_features*4, 1, stride=1, bias=False),
                nn.BatchNorm2d(out_features*4)
                )
        else:
            stride = 2
            self.identity = nn.Sequential(
                nn.Conv2d(in_features, out_features*4, 1, stride=2, bias=False),
                nn.BatchNorm2d(out_features*4)
                )

        self.batchnorm1 = nn.BatchNorm2d(in_features)
        self.conv1 = nn.Conv2d(in_features, out_features, 1, stride=stride, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_features)
        self.conv2 = nn.Conv2d(out_features, out_features, 3, stride=1, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(out_features)
        self.conv3 = nn.Conv2d(out_features, out_features*4, 1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.batchnorm1(x)
        y = self.relu(y)
        y = self.conv1(y)
        y = self.batchnorm2(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.batchnorm3(y)
        y = self.relu(y)
        y = self.conv3(y)
        y += self.identity(x)
        
        return y

class PreactResNet18_fewshot(nn.Module):
    def __init__(self, num_classes, dataset):
        super(PreactResNet18_fewshot, self).__init__()
        
        self.block_type = 'Residual'
        layers = [2,2,2,2]
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
        )
        self.conv2_x = self.build_blocks(layers[0], 64, 64)
        self.conv3_x = self.build_blocks(layers[1], 64, 160)
        self.conv4_x = self.build_blocks(layers[2], 160, 320)
        self.conv5_x = self.build_blocks(layers[3], 320, 640)
        if dataset == 'miniimagenet':
            self.avgpool = nn.AvgPool2d(8)
        else:
            self.avgpool = nn.AvgPool2d(4)
        self.relu = nn.ReLU()
        if self.block_type == 'Bottleneck':
            self.fc = nn.Linear(2048, num_classes)
        else:
            self.fc = nn.Linear(640, num_classes)
            
    def build_blocks(self, layer, in_features, out_features):
        module_list = []
        if self.block_type == 'Residual':
            for i in range(layer):
                if i == 0:
                    module_list.append(PreactResidualBlock(in_features, out_features))
                else:
                    module_list.append(PreactResidualBlock(out_features, out_features))
        elif self.block_type == 'Bottleneck':
            for i in range(layer):
                if i == 0:
                    module_list.append(PreactBottleneckBlock(in_features, out_features))
                else:
                    module_list.append(PreactBottleneckBlock(out_features*4, out_features))
        return nn.Sequential(*module_list)
            
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        return x

class PreactResNetBackbone(nn.Module):
    def __init__(self, original_model):
        super(PreactResNetBackbone, self).__init__()
        self.conv1 = original_model.conv1
        self.conv2_x = original_model.conv2_x
        self.conv3_x = original_model.conv3_x
        self.conv4_x = original_model.conv4_x
        self.conv5_x = original_model.conv5_x
        self.relu = original_model.relu
        self.avgpool = original_model.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x 

def set_net(model, num_classes, dataset):
    if model == 'ResNet-18':
        net = PreactResNet18_fewshot(num_classes, dataset)
    return net