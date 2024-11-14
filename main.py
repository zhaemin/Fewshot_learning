import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import torch.nn.utils as utils

import dataloader

from utils import parsing_argument, load_model, set_parameters

from torch.utils.tensorboard import SummaryWriter


def train_per_epoch(args, dataloader, net, optimizer, device):
    net.train()
    
    running_loss = 0.0
    tasks = []
    
    for data in dataloader:
        inputs, labels = data
        
        inputs, labels = inputs.to(device), labels.to(device)
        loss, acc = net(args, inputs, labels, args.train_num_ways, device)
        
        if loss == None:
            continue
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()
    
    return running_loss


def test(args, testloader, net, device):
    total = 0
    correct = 0
    total_loss = 0
    total_acc = 0
    
    net.eval()
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        if args.model == 'maml':
            loss, acc = net(args, inputs, labels, args.test_num_ways, device)
        else:
            with torch.no_grad():
                loss, acc = net(args, inputs, labels, args.test_num_ways, device)
        
        total_acc += acc
    accuracy = total_acc/len(testloader)
    
    return loss, accuracy

def train(args, trainloader, validloader, net, optimizer, scheduler, device, writer, outputs_log):
    max_val_acc = -1
    patience = 0
    
    for epoch in range(args.epochs):
        running_loss = train_per_epoch(args, trainloader, net, optimizer, device)
        
        print('epoch[%d] - training loss : %.3f'%(epoch+1, running_loss))
        print('epoch[%d] - training loss : %.3f'%(epoch+1, running_loss), file=outputs_log)
        writer.add_scalar('train / train_loss', running_loss, epoch+1)
        
        torch.save(net.state_dict(), './moco.pt')
        
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
        
    print('Training finished',file=outputs_log)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = parsing_argument()
    cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    
    outputs_log = open(f'outputs/{args.unsupervised}_{args.model}_{args.epochs}ep_{args.learningrate}lr_{args.optimizer}_{args.test_num_ways}ways_{args.num_shots}shots_{cur_time}.txt','w')
    writer = SummaryWriter(f'logs/{args.unsupervised}_{args.model}_{args.epochs}ep_{args.learningrate}lr_{args.optimizer}_{args.test_num_ways}ways_{args.num_shots}shots_{cur_time}')
    
    trainloader,testloader, validloader = dataloader.load_dataset(args, device)
    
    net = load_model(args)
    net.to(device)
    #net.load_state_dict(torch.load('./moco_pretext.pt'))
    optimizer,scheduler = set_parameters(args, net)
    
    if args.train:
        print(f"Training start ...")
        train(args, trainloader, validloader, net, optimizer, scheduler, device, writer, outputs_log)
    if args.test:
        print(f"Test start ...")
        #net.load_state_dict(torch.load('./model_state_dict_fewshot.pt'))
        loss,acc = test(args, testloader, net, device)
        print("accuracy: %.3f %%"%acc)
        print("accuracy: %.3f %%"%acc, file=outputs_log)
        
    outputs_log.close()
    writer.close()

if __name__ == "__main__":
    main()
