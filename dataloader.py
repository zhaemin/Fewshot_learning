import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler

import models.cactus as cactus

'''
src : https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch
'''

class FewShotSampler(Sampler):
    def __init__(self, labels, num_ways, num_shots, num_queries, episodes, num_tasks, data_source=None):
        super().__init__(data_source)
        self.labels = labels
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_queries = num_queries
        self.episodes = episodes
        self.num_tasks = num_tasks
        
        self.classes, self.counts = torch.unique(self.labels, return_counts=True)
        self.num_classes = len(self.classes)
        self.data_matrix = torch.Tensor(np.empty((len(self.classes), max(self.counts)), dtype=int)*np.nan)
        self.num_per_class = torch.zeros_like(self.classes)
        
        #data_matrix => 해당 class에 맞는 데이터의 index를 저장
        #np.where => nan인 값들이 2차원으로 반환됨 [[nan, nan, ..., nan]]
        
        for data_idx, label in enumerate(labels):
            self.data_matrix[label, np.where(np.isnan(self.data_matrix[label]))[0][0]] = data_idx 
            self.num_per_class[label] += 1
        
        self.valid_classes = [c.item() for c, count in zip(self.classes, self.num_per_class) if count >= self.num_shots+self.num_queries]
        
    def __iter__(self):
        for _ in range(self.episodes):
            tasks = []
            for t in range(self.num_tasks):
                batch_support_set = torch.LongTensor(self.num_ways*self.num_shots)
                batch_query_set = torch.LongTensor(self.num_ways*self.num_queries)
                
                way_indices = torch.randperm(len(self.valid_classes))[:self.num_ways]
                selected_classes = [self.valid_classes[idx] for idx in way_indices]
                
                for i, label in enumerate(selected_classes):
                    slice_for_support = slice(i*self.num_shots, (i+1)*self.num_shots)
                    slice_for_queries = slice(i*self.num_queries, (i+1)*self.num_queries)
                    
                    samples = torch.randperm(self.num_per_class[label])[:self.num_shots+self.num_queries]
                    batch_support_set[slice_for_support] = self.data_matrix[label][samples][:self.num_shots]
                    batch_query_set[slice_for_queries] = self.data_matrix[label][samples][self.num_shots:]
                
                batch = torch.cat((batch_support_set, batch_query_set))
                tasks.append(batch)
            
            batches = torch.cat(tasks)
            yield batches
            
    def __len__(self):
        return self.episodes

class UMTRADatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform_train = transforms.PILToTensor()
        self.normalize = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        self.autoaugment = transforms.AutoAugment()
        self.targets = torch.arange(0, len(dataset), 1 / 2).long()
        
    def __len__(self):
        return len(self.dataset)*2
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx//2]
        x = self.transform_train(x)
        
        if idx % 2 == 1: 
            x = self.autoaugment(x)
        
        x = self.normalize(x/255.0)
        y = idx // 2
        
        return x, y

class CACTUsDatasetWrapper(Dataset):
    def __init__(self, dataset, partition):
        self.dataset = dataset
        self.labels = partition
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        y = self.labels[idx]
        
        return x, y

class MoCoDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([ 
            transforms.RandomResizedCrop((32,32)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x1 = self.transform(x)
        x2 = self.transform(x)
        x = torch.stack((x1,x2))
        
        return x, y

def load_dataset(args, device):
    transform_test = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
    
    if args.dataset == 'miniimagenet':
        transform_train = transforms.Compose([ 
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((84,84), antialias=True),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
        
        trainset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/miniimagenet/data/train', transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/miniimagenet/data/test', transform=transform_test)
        valset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/miniimagenet/data/val', transform=transform_test)
        
    elif args.dataset == 'cifarfs':
        transform_train = transforms.Compose([ 
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((32,32)),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
        
        trainset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/cifar100/data/train_data', transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/cifar100/data/test_data', transform=transform_test)
        valset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/cifar100/data/val_data', transform=transform_test)
    
    elif args.dataset == 'cifar10':
        transform_train = transforms.Compose([ 
            transforms.ToTensor(),
            transforms.RandomResizedCrop((32,32)),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
        
        transform_test = transforms.Compose([ 
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
        
        trainset = torchvision.datasets.CIFAR10(root = '../data/cifar10', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root = '../data/cifar10', train=False, download=True, transform=transform_test)
        valset = torchvision.datasets.CIFAR10(root = '../data/cifar10', train=False, download=True, transform=transform_test)
    
    trainset_labels = torch.LongTensor(trainset.targets)
    testset_labels = torch.LongTensor(testset.targets)
    valset_labels = torch.LongTensor(valset.targets)
    
    train_sampler = FewShotSampler(trainset_labels, args.train_num_ways, args.num_shots, args.num_queries, args.episodes, args.num_tasks)
    test_sampler = FewShotSampler(testset_labels, args.test_num_ways, args.num_shots, args.num_queries, 600, num_tasks=1)
    val_sampler = FewShotSampler(valset_labels, args.test_num_ways, args.num_shots, args.num_queries, 100, num_tasks=1)
    
    if args.unsupervised == 'umtra':
        trainset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/miniimagenet/data/train')
        trainset = UMTRADatasetWrapper(trainset)
        train_sampler = FewShotSampler(trainset_labels, args.train_num_ways, 1, 1, args.episodes, args.num_tasks)

    elif args.unsupervised == 'cactus':
        trainset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/miniimagenet/data/train', transform=transform_test)
        train_loader = DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
        trainset_labels = cactus.make_cactus_paritions(train_loader, 500, device)
        trainset = CACTUsDatasetWrapper(trainset, trainset_labels)
        train_sampler = FewShotSampler(trainset_labels, args.train_num_ways, args.num_shots, args.num_queries, args.episodes, args.num_tasks)
    
    train_loader = DataLoader(trainset, batch_sampler=train_sampler, pin_memory=True)
    test_loader = DataLoader(testset, batch_sampler=test_sampler, pin_memory=True)
    val_loader = DataLoader(valset, batch_sampler=val_sampler, pin_memory=True)
    
    if args.model == 'moco':
        if args.pretext:
            trainset = torchvision.datasets.CIFAR10(root = '../data/cifar10', train=True, download=True)
            trainset = MoCoDatasetWrapper(trainset)
        train_loader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
        test_loader = DataLoader(testset, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
    
    return train_loader, test_loader, val_loader