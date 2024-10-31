import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import torch.nn.utils as utils

import dataloader
from nets.resnet12 import Res12

from utils import split_support_query_set, parsing_argument, load_model, set_parameters

from torch.utils.tensorboard import SummaryWriter


def train_per_epoch(args, dataloader, net, optimizer, device):
    net.train()
    running_loss = 0.0

    for i,data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs, True)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss


def test(args, testloader, net, device):
    total = 0
    correct = 0
    total_loss = 0
    
    net.eval()
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        tasks = split_support_query_set(inputs, labels, args.test_num_ways, args.num_shots, args.num_queries, 1, device)
        x_support, x_query, y_support, y_query = tasks[0]
        with torch.no_grad():
            support_set = net(x_support, False)
            query_set = net(x_query, False)
            prototypes = torch.mean(support_set.view(args.test_num_ways, -1, 640), dim=1)
            logits = -torch.cdist(query_set, prototypes)*(1/64)
            
            loss = F.cross_entropy(logits, y_query)
            _, predicted = torch.max(logits.data,1)
            total += 15 * args.test_num_ways
            correct += (predicted == y_query).sum().item()
    accuracy = 100*correct/total
    
    return loss, accuracy

def train(args, trainloader, validloader, net, optimizer, scheduler, device, writer, outputs_log):
    max_val_acc = -1
    patience = 0
    for epoch in range(args.epochs):
        running_loss = train_per_epoch(args, trainloader, net, optimizer, device)
        
        print('epoch[%d] - training loss : %.3f'%(epoch+1, running_loss/1000))
        print('epoch[%d] - training loss : %.3f'%(epoch+1, running_loss/1000), file=outputs_log)
        writer.add_scalar('train / train_loss', running_loss/1000, epoch+1)
        
        running_loss = 0.0
        
        if args.val:
            _, val_acc = test(args, validloader, net, device)
            print('          - validation acc : %.3f'%(val_acc))
            print('          - validation acc : %.3f'%(val_acc), file=outputs_log)
            writer.add_scalar('train / val_acc', val_acc, epoch+1)
            
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                torch.save(net.state_dict(), './res12_fewshot_pretrained.pt')
                patience = 0
            else:
                patience += 1
                print(f'patience: {patience}')
            if epoch>= 100 and patience >= 10:
                break
            
        if scheduler:
            scheduler.step()
        
    print('Finished Training',file=outputs_log)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parsing_argument()
    cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    
    outputs_log = open(f'outputs/{args.model}_{args.epochs}ep_{args.learningrate}lr_{args.optimizer}_{args.test_num_ways}ways_{args.num_shots}shots_{cur_time}.txt','w')
    writer = SummaryWriter(f'logs/{args.model}_{args.epochs}ep_{args.learningrate}lr_{args.optimizer}_{args.test_num_ways}ways_{args.num_shots}shots_{cur_time}')

    trainloader,testloader, validloader = dataloader.load_dataset(args)
    
    net = Res12()
    net.to(device)
    net.load_state_dict(torch.load('./pretrained_resnet12.pt'))
    optimizer,scheduler = set_parameters(args, net)
    
    if args.train:
        print(f"Training start ...")
        train(args, trainloader, validloader, net, optimizer, scheduler, device, writer, outputs_log)
    if args.test:
        print(f"Test start ...")
        net.load_state_dict(torch.load('./res12_fewshot_pretrained.pt'))
        loss,acc = test(args, testloader, net, device)
        print("accuracy: %.3f %%"%acc)
        print("accuracy: %.3f %%"%acc, file=outputs_log)
        
    outputs_log.close()
    writer.close()

if __name__ == "__main__":
    main()
