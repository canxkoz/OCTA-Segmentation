#### prediction
import os,sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import glob
from PIL import Image
from os import listdir

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models

import random
import math
import pytorch_unet
from torchvision.utils import save_image
import torch.nn.functional as F
from statistics import mean


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")

    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

fold = 2
run = 1
model_name = 'UNet_fold{}_run{}'.format(fold, run)

device = torch.device('cuda')
dtype = torch.cuda.FloatTensor
model = pytorch_unet.UNet(1).to(device)
model.load_state_dict(torch.load('./models/' + model_name))
model.eval()   # Set model to evaluate mode

ori_path = '/kuacc/users/hpc-ckoz/data/ROSE-2/test/original/'
gt_path = '/kuacc/users/hpc-ckoz/data/ROSE-2/test/gt/'
save_dir = 'UnetRoseResults/'
mkdir('UnetRoseResults')
imgs = os.listdir(ori_path)

#

auc_lst = []
acc_lst = []
sen_lst = []
fdr_lst = []
spe_lst = []
kappa_lst = []
gmean_lst = []
iou_lst = []
dice_lst = []
from evaluation import *
#

for img_path in imgs:
    if not img_path.endswith('.png'):
        continue

    simple_transform = transforms.ToTensor()
    
    ori = Image.open(ori_path + img_path).resize((512,512))
    ori = ori.convert("L")
    ori = simple_transform(ori).unsqueeze(0)
    pred = model(ori.to(device))
    pred = torch.sigmoid(pred)
    # pred = F.softmax(pred, dim = 1)
    pred = pred.data.cpu().numpy()
    pred = pred[0,0]
    pred_img = np.array(pred * 255, np.uint8)
    _, img = cv2.threshold(pred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    pink = (255,0,255)
    img = np.expand_dims(img, 2)
    img = np.repeat(img, 3, 2)
    white_pixels = np.where(
        (img[:, :, 0] == 255) & 
        (img[:, :, 1] == 255) & 
        (img[:, :, 2] == 255)
    )
    img[white_pixels] = pink

    gt = cv2.imread(gt_path + img_path)

    res = np.zeros((512, 512*3, 3))
    res[:,:512,:] = gt
    res[:,512:2*512, :] = cv2.resize(cv2.imread(ori_path + img_path), (512,512))
    res[:, 2*512:, :] = img
    
    
    ###########
    thresh_pred_img = img
    gt_img = res[:,:512,:]
    auc_lst.append(calc_auc(thresh_pred_img / 255.0, gt_img / 255.0))
    acc_lst.append(calc_acc(thresh_pred_img / 255.0, gt_img / 255.0))
    sen_lst.append(calc_sen(thresh_pred_img / 255.0, gt_img / 255.0))
    fdr_lst.append(calc_fdr(thresh_pred_img / 255.0, gt_img / 255.0))
    spe_lst.append(calc_spe(thresh_pred_img / 255.0, gt_img / 255.0))
    kappa_lst.append(calc_kappa(thresh_pred_img / 255.0, gt_img / 255.0))
    gmean_lst.append(calc_gmean(thresh_pred_img / 255.0, gt_img / 255.0))
    iou_lst.append(calc_iou(thresh_pred_img / 255.0, gt_img / 255.0))
    dice_lst.append(calc_dice(thresh_pred_img / 255.0, gt_img / 255.0))
  

    #########
    
    
    
    cv2.imwrite(save_dir + img_path, res.astype(int))
    
######
print("--------------- ACC List---------------")
print(acc_lst)
print("--------------- SEN List---------------")
print(sen_lst)
print("--------------- FDR List---------------")
print(fdr_lst)
print("--------------- AUC List---------------")
print(auc_lst)
print("--------------- SPE List---------------")
print(spe_lst)
print("--------------- KAPPA List---------------")
print(kappa_lst)
print("--------------- GMEAN List---------------")
print(gmean_lst)
print("--------------- IOU List---------------")
print(iou_lst)
print("--------------- DICE List---------------")
print(dice_lst)

print("UNET KERNEL SIZE (3,3)")

print("--------------- ACC Mean---------------")
print(mean(acc_lst))
print("--------------- SEN Mean---------------")
print(mean(sen_lst))
print("--------------- FDR Mean---------------")
print(mean(fdr_lst))
print("--------------- AUC Mean---------------")
print(mean(auc_lst))
print("--------------- SPE Mean---------------")
print(mean(spe_lst))
print("--------------- KAPPA Mean---------------")
print(mean(kappa_lst))
print("--------------- GMEAN Mean---------------")
print(mean(gmean_lst))
print("--------------- IOU Mean---------------")
print(mean(iou_lst))
print("--------------- DICE Mean---------------")
print(mean(dice_lst))

#######