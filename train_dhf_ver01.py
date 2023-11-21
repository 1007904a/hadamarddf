import os
import time 
import models
import shutil
import collections
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from scipy.linalg import hadamard

import warnings
warnings.filterwarnings('ignore')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(predictions, targets, codes):
    res = [0.0, 0.0]
    
    criterion = nn.MSELoss().cuda()
    for idx, pred in enumerate(predictions):
        errors = np.zeros(len(codes[0]))
            
        for j in range(0, len(errors)):
            errors[j] = criterion(pred, codes[j])

        error_sort = errors.argsort()[:5]

        if targets[idx].item() in error_sort[:1]:
            res[0] += 1.0

        if targets[idx].item() in error_sort:
            res[1] += 1.0
            
        
    return (res[0]/predictions.size(0))*100.0, (res[1]/predictions.size(0))*100.0


def train(train_loader, model, criterion, optimizer, codes):
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    model.train()
    for i, (images, target) in enumerate(train_loader):
        images  = images.cuda()
        targets = codes[target]
        
        # compute predictions
        predictions = model(images)
        loss        = criterion(predictions, targets)
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(predictions, target, codes)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            

    return [round(losses.avg, 4), round(top1.avg, 2), round(top5.avg, 2)]


def validation(val_loader, model, criterion, codes):
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()
    for i, (images, target) in enumerate(val_loader):
        images  = images.cuda()
        targets = codes[target]
        
        # compute output
        predictions = model(images)
        loss        = criterion(predictions, targets)

        acc1, acc5 = accuracy(predictions, target, codes)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))
        
    
    return [round(losses.avg, 4), round(top1.avg, 2), round(top5.avg, 2)]


def main(model_name='ResNet50', codes_size=64, epochs=50, img_size=224, batch_size=512, filename=''):
    #Data loading code
    mainPath = '/media/ajhoyos/OS/Datasets/Mini-ImageNet-New/Full_256/'
    
    traindir = os.path.join(mainPath, 'train')
    valdir   = os.path.join(mainPath, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    data_transformation = transforms.Compose([transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                                              transforms.CenterCrop(size=img_size),
                                              transforms.RandomRotation(15),
                                              transforms.ColorJitter(),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(), normalize])

    train_data = datasets.ImageFolder(traindir, data_transformation)
    val_data   = datasets.ImageFolder(valdir, transforms.Compose([transforms.CenterCrop(size=img_size), transforms.ToTensor(), normalize]))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=False)
    val_loader   = torch.utils.data.DataLoader(val_data,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    
    
    #Define codes
    codes = hadamard(codes_size)
    codes = torch.tensor(codes, dtype=torch.float).cuda()
    
    
    #Define the model
    if model_name == 'ResNet18': 
        model = models.ResNet18()
    elif model_name == 'ResNet50': 
        model = models.ResNet50()
    elif model_name == 'ResNet101': 
        model = models.ResNet101()
    elif model_name == 'ResNet152': 
        model = models.ResNet152()
        
        
    for param in model.parameters():
        param.requires_grad = False
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(codes[0]))
    
    #model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    cudnn.benchmark = True
    
    #Define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters())

    torch.set_num_threads(2)
    
    best_prec1_val = -1
    for epoch in range(0, epochs):
        train_data = train(train_loader, model, criterion, optimizer, codes)     
        val_data   = validation(val_loader, model, criterion, codes)
        
        print('Epoch{0}\t\tTrain loss:{1:.6f}\tTop-1:{2:.2f}\tTop-5:{3:.2f}\tVal loss:{4:.6f}\tTop-1:{5:.2f}\tTop-5:{6:.2f}'.format(epoch+1, train_data[0], train_data[1], train_data[2], val_data[0], val_data[1], val_data[2]))
        
        is_best_val = val_data[1] >= best_prec1_val
        
        if is_best_val:
            best_prec1_val = val_data[1]
            state = {'epoch':epoch + 1, 'arch':model_name, 'state_dict':model.state_dict()}
            torch.save(state, './checkpoint/' + model_name + '/'+ model_name + filename + '.pth')
            
    
    print('\n', model_name, 'done!')
    torch.cuda.empty_cache()


if __name__ == "__main__":
	main(epochs=100, batch_size=4, filename='_c11_lfmse_hardtanh_test')

