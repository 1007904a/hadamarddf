import numpy as np
from PIL import Image

import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torchvision import models

# base dir from proyect ... 
base_dir = os.getcwd()  #'/media/eduardo/ADATA HV300/images_df'

# input size image 
input_size = 224

# batch size
batch_size = 32

#
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

def load_model(model_name, num_cls=-1, ef=-1, pretrained='IMAGENET1K_V1'):
    print("{}".format(model_name))

    # load model from torchvision.models ... 
    if   ef == 0:
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    elif ef == 1:
        model = models.efficientnet_b1(weights='IMAGENET1K_V1')
    elif ef == 2:
        model = models.efficientnet_b2(weights='IMAGENET1K_V1')
    elif ef == 3:
        model = models.efficientnet_b3(weights='IMAGENET1K_V1')

    elif ef == 4:
        model = models.resnet50(weights='IMAGENET1K_V1')  # IMAGENET1K_V2
    elif ef == 5:
        model = models.resnet101(weights='IMAGENET1K_V1') # IMAGENET1K_V2

    elif ef == 6:
        model = models.vgg16(weights='IMAGENET1K_V1')

    elif ef == 7: 
        model = models.mnasnet1_3(weights='IMAGENET1K_V1')

    elif ef == 8:
        model = models.convnext_large(weights='IMAGENET1K_V1')

    elif ef == 9:
        model = models.vit_h_14(weights='IMAGENET1K_SWAG_E2E_V1')

    elif ef == 10:
        model = models.regnet_y_128gf(weights='IMAGENET1K_SWAG_E2E_V1')

    elif ef == 11:
        model = models.maxvit_t(weights='IMAGENET1K_V1')

    elif ef == 12:
        model = models.swin_v2_b(weights='IMAGENET1K_V1')

    # freezing cnn model
    for param in model.parameters():
        param.requires_grad = False


    # drop the last layer before model's output
    # == Efficient Net ==
    if   ef in [0, 1, 2, 3]:
        model.classifier = model.classifier[:-1]

    # == Resnet ==
    elif ef in [4, 5]:
        model = torch.nn.Sequential(*(list(model.children())[:-1]))

    # == VGG, mnasnet1_3, convnext_large, maxvit_t ==
    elif ef in [6, 7, 8, 11]:
        model.classifier = model.classifier[:-1]

    # === vit_h_14, regnet_y_128gf, swin_v2_b ===
    elif ef in [9, 10, 12]:
        model = torch.nn.Sequential(*(list(model.children())[:-1]))

    model.to(device)
    model.eval()

    return model


loader = transforms.Compose([transforms.RandomResizedCrop(size=input_size, scale=(0.8, 1.0)),
                             transforms.CenterCrop(size=input_size),
                             transforms.RandomRotation(15),
                             transforms.ColorJitter(),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(), 
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def image_loader(image_path):
    image = Image.open(image_path)
    image = loader(image).float()
    image = image.unsqueeze(0) 
    return image.to(device)


def main(model_class, files_dir, ef):
    
    model = load_model(model_name="", num_cls=0, ef=ef, pretrained=True)

    image_datasets = datasets.ImageFolder(files_dir, loader) 

    dataloader_dict = torch.utils.data.DataLoader(image_datasets, batch_size=(batch_size), shuffle=False, num_workers=16) 

    # create a dir to save data ... 
    txt_dir = "{}/corr_mtrx_00_{}".format(base_dir, model_class)
    try:
        os.system("rm -rf {}".format(txt_dir))
    except:
        pass
    os.system("mkdir  {}".format(txt_dir))


    # to extract data ... 
    for _dd_idx, (imgs, labels) in enumerate(dataloader_dict):

        imgs = imgs.to(device)
        outputs = model(imgs)

        labels = labels.numpy()

        for i, output in enumerate(outputs):
            out_ = output.detach().cpu().numpy().ravel()
            out2_ = np.array(out_, dtype=str)

            txt_classes_files =  "{}/txt_classes.txt".format(txt_dir)
            txt_vectores_files = "{}/txt_vectors.txt".format(txt_dir)

            with open(txt_classes_files, 'a+') as f2:
                f2.write("{}\n".format(labels[i]))
            f2.close()

            with open(txt_vectores_files, 'a+') as f1:
                f1.write("{}\n".format(",".join(out2_)))
            f1.close()



if __name__ == "__main__":

    # route to databaset, e.g. ImageNet
    data_base  = "imagenet"
    route_val_   = "/media/eduardo/963A11D33A11B0EB/phd/data/ILSVRC/Data/CLS-LOC/val"
    route_train_ = "/media/eduardo/963A11D33A11B0EB/phd/data/ILSVRC/Data/CLS-LOC/train"

    """
    IDX | NAME
    0   | ef0
    ...
    12  | swin_v2_b
    """
    models_name = ["ef0", "ef1", "ef2", "ef3", "resnet50", "resnet101", "vgg16", "mnasnet1_3", "convnext_large", "vit_h_14", "regnet_y_128gf", "maxvit_t", "swin_v2_b"]
    models_idx  = [    0,    1,     2,     3,           4,           5,       6,            7,                8,          9,               10,         11,          12]

    for idx in [0, 1, 2, 3]: # use idxs from 'models_idx'
        base_model = "df_{}".format(models_name[idx])

        dir_files = route_val_
        main("{}_{}_val  ".format(base_model, data_base), dir_files, ef=idx)

        dir_files = route_train_
        main("{}_{}_train".format(base_model, data_base), dir_files, ef=idx)
