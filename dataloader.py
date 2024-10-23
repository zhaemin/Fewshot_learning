import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Sampler

'''
src : https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch
'''

class FewShotSampler(Sampler):
    def __init__(self, labels, num_ways, num_shots, num_queries, episodes, data_source=None):
        super().__init__(data_source)
        self.labels = labels
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_queries = num_queries
        self.episodes = episodes
        
        self.classes, self.counts = torch.unique(self.labels, return_counts=True)
        self.num_classes = len(self.classes)
        self.data_matrix = torch.Tensor(np.empty((len(self.classes), max(self.counts)), dtype=int)*np.nan)
        self.num_per_class = torch.zeros_like(self.classes)
        
        '''
        data_matrix => 해당 class에 맞는 데이터의 index를 저장
        np.where => nan인 값들이 2차원으로 반환됨 [[nan, nan, ..., nan]]
        '''
        for data_idx, label in enumerate(labels):
            self.data_matrix[label, np.where(np.isnan(self.data_matrix[label]))[0][0]] = data_idx 
            self.num_per_class[label] += 1
            
    def __iter__(self):
        for _ in range(self.episodes):
            batch_support_set = torch.LongTensor(self.num_ways*self.num_shots)
            batch_query_set = torch.LongTensor(self.num_ways*self.num_queries)
            
            way_indices = torch.randperm(self.num_classes)[:self.num_ways]
            
            for i, label in enumerate(way_indices):
                slice_for_support = slice(i*self.num_shots, (i+1)*self.num_shots)
                slice_for_queries = slice(i*self.num_queries, (i+1)*self.num_queries)
                
                samples = torch.randperm(self.num_per_class[label])[:self.num_shots+self.num_queries]
                batch_support_set[slice_for_support] = self.data_matrix[label][samples][:self.num_shots]
                batch_query_set[slice_for_queries] = self.data_matrix[label][samples][self.num_shots:]
            
            batch = torch.cat((batch_support_set, batch_query_set))
            yield batch
            
    def __len__(self):
        return self.episodes

def load_dataset(dataset, num_ways, num_shots, num_queries, episodes):
    transform_test = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
    
    if dataset == 'miniimagenet':
        transform_train = transforms.Compose([ 
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((84,84)),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
        
        trainset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/miniimagenet/data/train', transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/miniimagenet/data/test', transform=transform_test)
        valset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/miniimagenet/data/val', transform=transform_test)
        num_train_labels = 64
        
    elif dataset == 'cifarfs':
        transform_train = transforms.Compose([ 
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((32,32)),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
        
        trainset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/cifar100/data/train_data', transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/cifar100/data/test_data', transform=transform_test)
        valset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/cifar100/data/val_data', transform=transform_test)
        num_train_labels = 64
        
    trainset_labels = torch.LongTensor(trainset.targets)
    testset_labels = torch.LongTensor(testset.targets)
    valset_labels = torch.LongTensor(valset.targets)
    
    train_sampler = FewShotSampler(trainset_labels, num_ways, num_shots, num_queries, episodes)
    test_sampler = FewShotSampler(testset_labels, num_ways, num_shots, 15, 600)
    val_sampler = FewShotSampler(valset_labels, num_ways, num_shots, 15, 100)
    
    train_loader = DataLoader(trainset, batch_sampler=train_sampler)
    test_loader = DataLoader(testset, batch_sampler=test_sampler)
    val_loader = DataLoader(valset, batch_sampler=val_sampler)
    
    return train_loader, test_loader, val_loader, num_train_labels
