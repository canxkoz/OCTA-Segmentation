import os,sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2 as cv
import glob
from PIL import Image
from os import listdir

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torch.utils.data as data
import torchvision.transforms.functional as TF

from torchsummary import summary
import torch.nn as nn
import pytorch_unet
import random 
from tqdm import tqdm

fold = 2
run = 1
reshape_size = 512
model_name = 'UNet_fold{}_run{}'.format(fold, run)
# makeDirectory('./results/' + model_name)

def build_dataset(data_dir, channel=1, isTraining=True, scale_size=(512, 512)):
    database = CRIA(data_dir, channel=channel, isTraining=isTraining, scale_size=scale_size)
    return database

# torch.cuda.set_device(1)

dataPath = './data/ROSE-2'
savePathPrefix = './'

trInputPath = dataPath + '/train/'
valInputPath = dataPath + '/validation/'
# tsInputPath = dataPath + '/test/'

outputPathtr = dataPath + '/golds/'
outputPathval = dataPath + '/golds/'
# outputPathts = ''

imagePostfix = '.png'
goldPostfix = '.tif'

class CRIA(data.Dataset):
    def __init__(self, root, channel=1, isTraining=True, scale_size=(512, 512)):
        super(CRIA, self).__init__()
        self.img_lst, self.gt_lst = self.get_dataPath(root, isTraining)
        self.channel = channel
        self.isTraining = isTraining
        self.scale_size = scale_size
        self.name = ""
        
        assert self.channel == 1 or self.channel == 3, "the channel must be 1 or 3"  # check the channel is 1 or 3
    
    def __getitem__(self, index):
        imgPath = self.img_lst[index]
        self.name = imgPath.split("/")[-1]
        gtPath = self.gt_lst[index]
        simple_transform = transforms.ToTensor()
        
        img = Image.open(imgPath)
        gt = Image.open(gtPath).convert("L")
       
        if self.channel == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        
        gt = np.array(gt)
        """gt[gt >= 128] = 255
        gt[gt < 128] = 0"""
        
        gt = Image.fromarray(gt)
    
        if self.isTraining:
            # augumentation
            rotate = 10
            angel = random.randint(-rotate, rotate)
            img = img.rotate(angel)
            gt = gt.rotate(angel)
        
        img = simple_transform(img)
        gt = simple_transform(gt)
        
        return img, gt, self.name
    
    def __len__(self):
        return len(self.img_lst)
    
    def get_dataPath(self, root, isTraining):
        if isTraining:
            img_dir = os.path.join(root + "/train/original")
            gt_dir = os.path.join(root + "/train/gt_dist")
        else:
            img_dir = os.path.join(root + "/test/original")
            gt_dir = os.path.join(root + "/test/gt_dist")
        
        img_lst = sorted(list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir))))
        gt_lst = sorted(list(map(lambda x: os.path.join(gt_dir, x), os.listdir(gt_dir))))
        
        assert len(img_lst) == len(gt_lst)
        
        return img_lst, gt_lst
    
    def getFileName(self):
        return self.name


def listAllOCTFiles(imageDirPath, imagePostfix):
    fileList = listdir(imageDirPath)
    postLen = len(imagePostfix)
    imageNames = []
    for i in range(len(fileList)):
        if fileList[i][-postLen::] == imagePostfix:
            imageNames.append(fileList[i][:-postLen])
    return imageNames


batch_size = 1
num_workers = 1
# Training data loader
train_dataset = build_dataset(dataPath, channel=1, isTraining=True)
val_dataset = build_dataset(dataPath, channel=1, isTraining=False)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

torch.autograd.set_detect_anomaly(True)

dataloaders = {
    'train': train_loader,
    'val': val_loader
}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

model = pytorch_unet.UNet(1)

# model= nn.DataParallel(model)
# torch.cuda.empty_cache()
model = model.to(device)

summary(model, input_size=(1, reshape_size, reshape_size))

import errno
def makeDirectory(directoryPath):
    try:
        os.mkdir(directoryPath) 
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    

from collections import defaultdict
import torch.nn.functional as F
# from loss import dice_loss

def calc_loss(pred, target, metrics, bce_weight=0.5):
#     bce = F.binary_cross_entropy_with_logits(pred, target, weights)
#     bce = F.binary_cross_entropy_with_logits(pred, target)
    #bce = nn.BCEWithLogitsLoss()(pred, target)
    mse = nn.MSELoss()(pred, target)
    rmse = torch.sqrt(mse)
    l1loss = nn.L1Loss()(pred, target)
#     pred = F.sigmoid(pred)
#     pred = F.softmax(pred, dim = 1)
#     dice = dice_loss(pred, target)
    
#     loss = bce * bce_weight + dice * (1 - bce_weight)
    loss = mse
#     metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
#     metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss


def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))   
    
    
dtype = torch.cuda.FloatTensor
# dtype = torch.FloatTensor

def train_model(model, optimizer, scheduler, patience, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e15

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
#                 scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0            
            for inputs, labels, names in tqdm(dataloaders[phase]):
                inputs = inputs.to(device).type(dtype)
                labels = labels.to(device).type(dtype)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss <= best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, './models/' + model_name)
                
            if phase == 'val':
                valid_loss = epoch_loss
                scheduler.step(epoch_loss)
                
#             model_wts = copy.deepcopy(model.state_dict())
#             torch.save(model_wts, './models/' + model_name + f'_{epoch}')

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model


def test_model(model, save_dir):
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    model.eval()   # Set model to evaluate mode

    metrics = defaultdict(float)
    epoch_samples = 0            
    for inputs, labels, names in tqdm(dataloaders['val']):
        inputs = inputs.to(device).type(dtype)
        labels = labels.to(device).type(dtype)

        outputs = model(inputs)
        for i,out in enumerate(outputs):
            np_out = out.detach().numpy()
            img = Image.fromarray(np_out[0], mode='L')
            img_name = names[i]
            img.save(os.path.join(save_dir, img_name))

        loss = calc_loss(outputs, labels, metrics)
        # statistics
        epoch_samples += inputs.size(0)

    print_metrics(metrics, epoch_samples, 'val')


import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy

num_class = 1

model = pytorch_unet.UNet(num_class)

# model= nn.DataParallel(model)   # device_ids=[1, 2]
model = model.to(device)
# model.load_state_dict(torch.load('./models/UNet_fold4_run1_40'))

optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)
# optimizer_ft = optim.Adadelta(filter(model.parameters()), lr=1e-1)

# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=60, gamma=0.1) 
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=10) 
#         
model = train_model(model, optimizer_ft, exp_lr_scheduler, patience = 30, num_epochs=100)

output_save_dir = './outputs/e20_mse'
model.load_state_dict(torch.load('./models/UNet_fold4_run_mse.pt'))
test_model(model, output_save_dir)