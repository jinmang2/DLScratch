import os

from PIL import Image
import numpy as np

import torch
import torch.nn as nn

import torch.nn.functional as F

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms



import argparse
from argparse import Namespace

from dataloader import dataloader
from model import ResNet
from traintools import train



def main(args):
    loaders = dataloader(args)
    model = ResNet(args.modelname, args.n_classes).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.learning_rate,
        momentum=args.momentum, 
        weight_decay=args.weight_decay)
    train()


if __name__ == '__main__':
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="학습할 지 여부!")
    parser.add_argument("--test_pct", type=float, default=0.2, help="test 비율")
    parser.add_argument("--val_pct", type=float, default=0.1, help="test 비율")
    parser.add_argument("--imgfolder", type=str, default='data/Image')
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_classes", type=int, default=120)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--modelname", type=str, default='resnet152')
    parser.add_argument("--verbose-epoch", type=int, default=0)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    print(type(args))

    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        torch.cuda.manual_seed_all(args.seed)
    else:
        args.device = torch.device('cpu')
        
    main(args)












