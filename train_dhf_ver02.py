from __future__ import print_function
from __future__ import division


#import models
from torchvision import datasets, models

import os
import copy
import time
import numpy as np
from scipy.linalg import hadamard 

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms


print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


import warnings
warnings.filterwarnings('ignore')

# Detect a GPU or CPU will be used 
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "mini-imagenet"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Number of classes in the dataset
num_classes = 100

# Batch size for training (change depending on how much memory you have)
batch_size = 4

# Number of epochs to train for
num_epochs = 100

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

# input size image 
input_size = 224

# hadamard codes ...
codes_size = 1024
codes = hadamard(codes_size)
idxs = np.where(codes <= 0)
for x, y in zip(idxs[0], idxs[1]):
    codes[x][y] = 0
codes = torch.tensor(codes, dtype=torch.float).to(device)

#
criterion = nn.MSELoss().to(device)

#
#model_ft = models.ResNet101(pretrained=True)
model_ft = models.resnet101(weights='IMAGENET1K_V1')

for param in model_ft.parameters():
    param.requires_grad = False

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(codes[0]))

print("Params to learn:")
params_to_update = model_ft.parameters()
params_to_update = []
for name, param in model_ft.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t",name)

# Send the model to GPU
model_ft = model_ft.to(device)

cudnn.benchmark = True

# Print the model we just instantiated
print(model_ft)

# optimizer
optimizer_ft = torch.optim.Adam(params_to_update)


print("Initializing Datasets and Dataloaders...")

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
             transforms.RandomResizedCrop(input_size),
             transforms.CenterCrop(input_size),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
           transforms.Resize(input_size),
           transforms.CenterCrop(input_size),
           transforms.ToTensor(),
           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create training and validation datasets
image_datasets   = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
                        for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) 
                        for x in ['train', 'val']}

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
        self.avg = round(self.sum / self.count, 4)

def accuracy(predictions, targets, codes):
    res = 0

    for idx, pred in enumerate(predictions):

        errors = list(map(lambda code: criterion(pred, code), codes))

        if device == "cuda:0":
            errors = [itm.data.cpu() for itm in errors]

        error_sort = np.array(errors).argsort()[:10]

        if   targets[idx].item() in error_sort[:1]:
            res += 1.0

    res = res / predictions.size(0)
    res *= 100
        
    return res


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    cudnn.benchmark = True
    torch.set_num_threads(2)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            top1 = AverageMeter()
            losses = AverageMeter()

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                #labels = labels.to(device)
                targets = codes[labels]

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    #loss = criterion(outputs, labels)

                    if phase == 'train':
                        acc1 = 0
                    else:
                        acc1 = accuracy(outputs, labels, codes)

                    losses.update(loss.item(), inputs.size(0))
                    top1.update(acc1, inputs.size(0))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, losses.avg, top1.avg))

            # deep copy the model
            if phase == 'val' and top1.avg > best_acc:
                best_acc = top1.avg
                best_model_wts = copy.deepcopy(model.state_dict())

                print("saving ... \n\n")
                torch.save(best_model_wts, 'checkpoint/resnet101_.pth')

    torch.cuda.empty_cache()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)

    return model


if __name__ == "__main__":
    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)
