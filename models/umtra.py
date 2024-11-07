import torch
import torchvision.transforms as transforms

def umtra_split(x, num_ways, device):
    x_list = torch.split(x[0].to(device), num_ways)
    transformed_x_list = torch.split(x[1].to(device), num_ways)
    
    tasks = []
    for x, transformed_x in zip(x_list, transformed_x_list):
        support_labels = torch.arange(0, num_ways).long().to(device)
        query_labels = torch.arange(0, num_ways).long().to(device)
        
        tasks.append([x, transformed_x, support_labels, query_labels])
    
    return tasks