import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim

import nets.resnet as resnet
import nets.resnet12 as resnet12
import nets.convnet as convnet

import models.protonet as protonet
import models.feat as feat
import models.relationnet as relationnet
import models.matchnet as matchnet
import models.maml as maml

def parsing_argument():
    parser = argparse.ArgumentParser(description="argparse_test")
    
    parser.add_argument('-e', '--epochs', metavar='int', type=int, help='epochs', default=2)
    parser.add_argument('-lr', '--learningrate', metavar='float', type=float, help='lr', default=0.0001)
    parser.add_argument('-d', '--dataset', metavar='str', type=str, help='dataset [miniimagenet, cifarfs]', default='miniimagenet')
    parser.add_argument('-opt', '--optimizer', metavar='str', type=str, help='optimizer [adam, sgd]', default='sgd')
    parser.add_argument('-crt', '--criterion', metavar='str', type=str, help='criterion [ce, mse]', default='ce')
    parser.add_argument('-tr', '--train', help='train', action='store_true')
    parser.add_argument('-tc', '--test', help='test', action='store_true')
    parser.add_argument('-val', '--val', help='validation', action='store_true')
    
    parser.add_argument('-tr_ways', '--train_num_ways', metavar='int', type=int, help='ways', default=5)
    parser.add_argument('-ts_ways', '--test_num_ways', metavar='int', type=int, help='ways', default=5)
    parser.add_argument('-shots', '--num_shots', metavar='int', type=int, help='shots', default=5)
    parser.add_argument('-tasks', '--num_tasks', metavar='int', type=int, help='tasks', default=1)
    parser.add_argument('-q', '--num_queries', metavar='int', type=int, help='queries', default=15)
    parser.add_argument('-ep', '--episodes', metavar='int', type=int, help='episodes', default=100)
    parser.add_argument('-m', '--model', metavar='str', type=str, help='models [protonet, feat, relationnet]', default='protonet')
    parser.add_argument('-b', '--backbone', metavar='str', type=str, help='backbone [res12, convnet]', default='res12')

    parser.add_argument('-usp', '--unsupervised', metavar='str', type=str, help='unsupervised learning-[umtra]', default=None)
    
    return parser.parse_args()

def load_model(args):
    if args.model != 'relationnet' and args.model != 'maml':
        if args.backbone == 'res12':
            backbone = resnet12.Res12()
            model_dict = backbone.state_dict()
            pretrained_dict = torch.load('res12_fewshot_pretrained.pt')
            emb_dim = 640
        elif args.backbone == 'convnet':
            backbone = convnet.ConvNet()
            #model_dict = backbone.state_dict()
            #pretrained_dict = torch.load('con_pre-avg.pth')['params']
            emb_dim = 64
        #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        #model_dict.update(pretrained_dict)
        #backbone.load_state_dict(model_dict)
        
        for param in backbone.parameters():
            param.requires_grad = True
        
    if args.model == 'feat':
        net = feat.FEAT(backbone, emb_dim)
    elif args.model == 'protonet':
        net = protonet.ProtoNet(backbone, emb_dim)
    elif args.model == 'relationnet':
        net = relationnet.RelationNet()
    elif args.model == 'matchnet':
        net = matchnet.MatchNet(backbone)
    elif args.model == 'maml':
        if args.unsupervised == 'umtra' or args.unsupervised == 'cactus':
            net = maml.MAML(inner_lr=0.05, num_ways=args.train_num_ways)
        else:
            net = maml.MAML(inner_lr=0.01, num_ways=args.train_num_ways)
    
    return net

def set_parameters(args, net):
    if args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr = args.learningrate)
        scheduler = None
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 80, 100, 120], gamma=0.5)
    if args.optimizer == 'sgd':
        if args.model == 'feat':
            optimizer = optim.SGD([
                {'params': net.embedding.parameters()}, 
                {'params': net.attention.parameters(), 'lr': args.learningrate*10}], 
                lr=args.learningrate, momentum=0.9, weight_decay=5e-4, nesterov=True)
        else:
            optimizer = optim.SGD(params=net.parameters(), lr=args.learningrate, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)
    
    return optimizer,scheduler

def split_support_query_set(x, y, num_class, num_shots, num_queries, num_tasks, device):
    
    x_list = torch.chunk(x, num_tasks)
    y_list = torch.chunk(y, num_tasks)
    tasks = []
    
    for i in range(num_tasks):
        x, y = x_list[i], y_list[i]
        num_sample_support = num_class * num_shots
        x_support, x_query = x[:num_sample_support], x[num_sample_support:]
        y_support, y_query = y[:num_sample_support], y[num_sample_support:]
        
        _classes = torch.unique(y_support)
        support_idx = torch.stack(list(map(lambda c: y_support.eq(c).nonzero(as_tuple=False).squeeze(1), _classes)))
        xs = torch.cat([x_support[idx_list] for idx_list in support_idx])
        
        query_idx = torch.stack(list(map(lambda c: y_query.eq(c).nonzero(as_tuple=False).squeeze(1), _classes)))
        xq = torch.cat([x_query[idx_list] for idx_list in query_idx])
        
        ys = torch.arange(0, len(_classes), 1 / num_shots).long().to(device)
        yq = torch.arange(0, len(_classes), 1 / num_queries).long().to(device)
        
        tasks.append([xs, xq, ys, yq])
        
    return tasks

