import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import torch.nn.utils as utils

import dataloader
import models.umtra as umtra
import models.cactus as cactus

from utils import split_support_query_set, parsing_argument, load_model, set_parameters

from torch.utils.tensorboard import SummaryWriter


def train_per_epoch(args, dataloader, net, task_generator, optimizer, device):
    net.train()
    
    running_loss = 0.0
    tasks = []
    
    if args.unsupervised == 'cactus':
        for i in range(args.episodes):
            tasks = task_generator.create_task_kmeans(args.num_tasks, args.train_num_ways, args.num_shots, args.num_queries)
            
            loss, outputs = net(tasks, args.train_num_ways)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
    else:
        for data in dataloader:
            inputs, labels = data
            
            if args.unsupervised == 'umtra':
                tasks = umtra.umtra_split(inputs, args.train_num_ways, device)
            else:
                inputs, labels = inputs.to(device), labels.to(device)
                tasks = split_support_query_set(inputs, labels, args.test_num_ways, args.num_shots, args.num_queries, args.num_tasks, device)
            
            loss, outputs = net(tasks, args.train_num_ways)
            
            optimizer.zero_grad()
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
        loss, outputs = net(tasks, args.train_num_ways)
        
        if args.model == 'maml':
            loss, outputs = net(tasks, args.train_num_ways)
        else:
            with torch.no_grad():
                loss, outputs = net(tasks, args.train_num_ways)
        
        with torch.no_grad():
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data,1)
            total += args.num_queries * args.train_num_ways
            correct += (predicted == y_query).sum().item()
    accuracy = 100*correct/total
    
    return loss, accuracy

def train(args, trainloader, validloader, net, optimizer, scheduler, device, writer, outputs_log):
    max_val_acc = -1
    patience = 0
    task_generator = None
    
    if args.unsupervised == 'cactus':
        task_generator = cactus.CACTUS(trainloader, num_partitions=1, device=device)
    
    for epoch in range(args.epochs):
        running_loss = train_per_epoch(args, trainloader, net, task_generator, optimizer, device)
        
        print('epoch[%d] - training loss : %.3f'%(epoch+1, running_loss))
        print('epoch[%d] - training loss : %.3f'%(epoch+1, running_loss), file=outputs_log)
        writer.add_scalar('train / train_loss', running_loss, epoch+1)
        
        running_loss = 0.0
        
        if args.val and (epoch+1)%10==0:
            _, val_acc = test(args, validloader, net, device)
            print('          - validation acc : %.3f'%(val_acc))
            print('          - validation acc : %.3f'%(val_acc), file=outputs_log)
            writer.add_scalar('train / val_acc', val_acc, epoch+1)
            
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                torch.save(net.state_dict(), './model_state_dict_fewshot.pt')
                patience = 0
            else:
                patience += 1
                print(f'patience: {patience}')
            #if patience >= 5:
            #    break
            
        if scheduler:
            scheduler.step()
        
    print('Finished Training',file=outputs_log)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parsing_argument()
    cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    
    outputs_log = open(f'outputs/{args.unsupervised}_{args.model}_{args.epochs}ep_{args.learningrate}lr_{args.optimizer}_{args.test_num_ways}ways_{args.num_shots}shots_{cur_time}.txt','w')
    writer = SummaryWriter(f'logs/{args.unsupervised}_{args.model}_{args.epochs}ep_{args.learningrate}lr_{args.optimizer}_{args.test_num_ways}ways_{args.num_shots}shots_{cur_time}')

    trainloader,testloader, validloader = dataloader.load_dataset(args)
    
    net = load_model(args)
    net.to(device)
    #net.load_state_dict(torch.load('./model_state_dict_fewshot.pt'))
    optimizer,scheduler = set_parameters(args, net)
    
    if args.train:
        print(f"Training start ...")
        train(args, trainloader, validloader, net, optimizer, scheduler, device, writer, outputs_log)
    if args.test:
        print(f"Test start ...")
        net.load_state_dict(torch.load('./model_state_dict_fewshot.pt'))
        loss,acc = test(args, testloader, net, device)
        print("accuracy: %.3f %%"%acc)
        print("accuracy: %.3f %%"%acc, file=outputs_log)
        
    outputs_log.close()
    writer.close()

if __name__ == "__main__":
    main()
