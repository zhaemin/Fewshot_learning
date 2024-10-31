import torch
import torchvision.transforms as transforms

def umtra_split(x, num_tasks, num_ways, device):
    x_list = torch.chunk(x, num_tasks)
    transformed_x_list = []
    
    for x in x_list:
        transformed_img_list = []
        for img in x:
            img_uint = (img * 255).to(torch.uint8)
            transform = transforms.AutoAugment()
            transformed_img = transform(img_uint).to(torch.float32)/255.0
            transformed_img_list.append(transformed_img)
        transformed_img_list = torch.stack(transformed_img_list)
        transformed_x_list.append(transformed_img_list)
    
    torch.stack(transformed_x_list)
    tasks = []
    
    for x, transformed_x in zip(x_list, transformed_x_list):
        support_labels = torch.arange(0, num_ways).long().to(device)
        query_labels = torch.arange(0, num_ways).long().to(device)
        
        tasks.append([x, transformed_x, support_labels, query_labels])
    
    return tasks