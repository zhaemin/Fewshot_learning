import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import torch.nn.utils as utils

import dataloader

from utils import split_support_query_set, parsing_argument, load_model, set_parameters

from torch.utils.tensorboard import SummaryWriter


def train_per_epoch(args, dataloader, net, optimizer, criterion, device):
    net.train()
    
    running_loss = 0.0
    tasks = []
    
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        x_support, x_query, y_support, y_query = split_support_query_set(inputs, labels, args.train_num_ways, args.num_shots, args.num_queries, device)
        
        if args.model == 'feat':
            outputs, aux_dist = net(x_support, x_query, args.train_num_ways)
            y = torch.arange(0, args.train_num_ways, 1/(args.num_shots+args.num_queries)).long().to(device)
            loss = criterion(outputs, y_query)+0.1*criterion(aux_dist, y)
        elif args.model == 'protonet':
            outputs = net(x_support, x_query, args.train_num_ways)
            loss = criterion(outputs, y_query)
        elif args.model == 'relationnet':
            one_hot_labels = F.one_hot(y_query).float().to(device)
            outputs = net(x_support, x_query, args.train_num_ways, args.num_shots, args.num_queries)
            loss = criterion(outputs, one_hot_labels)
        elif args.model == 'matchnet':
            outputs = net(x_support, x_query, y_support)
            loss = criterion(outputs, y_query)
        elif args.model == 'maml':
            if args.num_shots == 5:
                batch_size = 2
            else:
                batch_size = 4
                
            if len(tasks) == batch_size-1:
                tasks.append([x_support, x_query, y_support, y_query])
                loss, _ = net(tasks, num_inner_steps=5)
                running_loss+=loss
                tasks = []
            else:
                tasks.append([x_support, x_query, y_support, y_query])
                continue
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss


def test(args, testloader, net, criterion, device):
    total = 0
    correct = 0
    loss = 0
    
    net.eval()
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        x_support, x_query, y_support, y_query = split_support_query_set(inputs, labels, args.test_num_ways, args.num_shots, args.num_queries, device)
        
        with torch.no_grad():
            if args.model == 'feat':
                outputs = net(x_support, x_query, args.test_num_ways)
                tmp_loss = criterion(outputs, y_query)
            elif args.model == 'protonet':
                outputs = net(x_support, x_query, args.test_num_ways)
                tmp_loss = criterion(outputs, y_query)
            elif args.model == 'relationnet':
                one_hot_labels = F.one_hot(y_query).float().to(device)
                outputs = net(x_support, x_query, args.test_num_ways, args.num_shots, args.num_queries)
                tmp_loss = criterion(outputs, one_hot_labels)
            elif args.model == 'matchnet':
                outputs = net(x_support, x_query, y_support)
                tmp_loss = criterion(outputs, y_query)
                
        if args.model == 'maml':
            net.train()
            tmp_loss, outputs = net([[x_support, x_query, y_support, y_query]], num_inner_steps=10)
            net.eval()
            
        loss += tmp_loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data,1)
        total += y_query.size(0)
        correct += (predicted == y_query).sum().item()
    accuracy = 100*correct/total
    
    return loss, accuracy

def train(args, trainloader, validloader, net, optimizer, scheduler, criterion, device, writer, outputs_log):
    max_val_acc = -1
    patience = 0
    for epoch in range(args.epochs):
        running_loss = train_per_epoch(args, trainloader, net, optimizer, criterion, device)
        
        print('epoch[%d] - training loss : %.3f'%(epoch+1, running_loss))
        print('epoch[%d] - training loss : %.3f'%(epoch+1, running_loss), file=outputs_log)
        writer.add_scalar('train / train_loss', running_loss, epoch+1)
        
        running_loss = 0.0
        
        if args.val and (epoch+1)%5==0:
            _, val_acc = test(args, validloader, net, criterion, device)
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
            if patience >= 3:
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

    trainloader,testloader, validloader, num_classes_train = dataloader.load_dataset(args)
    
    net = load_model(args)
    net.to(device)
    #net.load_state_dict(torch.load('./model_state_dict_fewshot_relationnet1shot+100ep.pt'))
    optimizer,scheduler,criterion = set_parameters(args, net)
    
    if args.train:
        print(f"Training start ...")
        train(args, trainloader, validloader, net, optimizer, scheduler, criterion, device, writer, outputs_log)
    if args.test:
        print(f"Test start ...")
        net.load_state_dict(torch.load('./model_state_dict_fewshot.pt'))
        loss,acc = test(args, testloader, net, criterion, device)
        print("accuracy: %.3f %%"%acc)
        print("accuracy: %.3f %%"%acc, file=outputs_log)
        
    outputs_log.close()
    writer.close()

if __name__ == "__main__":
    main()
