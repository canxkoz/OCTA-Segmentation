import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from first_stage import SRF_UNet
from evaluation import *
    
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
    
folder_path = "/kuacc/users/hpc-ckoz/DRAC2022/OCTA"
pred_path = '/kuacc/users/hpc-ckoz/DRAC2022/OCTA/OCTA-Net-OCTA-Vessel-Segmentation-Network-master/code/OCTA-Net/results_new/cria/first_stage/fusion/Thresh/'
gt_path = '/kuacc/users/hpc-ckoz/data/ROSE-2/test/gt/'
ori_path = '/kuacc/users/hpc-ckoz/data/ROSE-2/test/original/'
save_dir = '/kuacc/users/hpc-ckoz/DRAC2022/OCTA/OctaRoseResults_pred/'
imgs = os.listdir(pred_path)
mkdir(folder_path + '/OctaRoseResults_pred')



auc_lst = []
acc_lst = []
sen_lst = []
fdr_lst = []
spe_lst = []
kappa_lst = []
gmean_lst = []
iou_lst = []
dice_lst = []


for img_path in imgs:
    if not img_path.endswith('.png'):
        continue
    img = cv2.imread(pred_path + img_path)
    pink = (255,0,255)
    white_pixels = np.where(
        (img[:, :, 0] == 255) & 
        (img[:, :, 1] == 255) & 
        (img[:, :, 2] == 255)
    )
    img[white_pixels] = pink

    gt = cv2.imread(gt_path + img_path)
    ori = cv2.imread(ori_path + img_path)

    res = np.zeros((512, 512*3, 3))
    res[:,0:512,:] = gt
    res[:,512:2*512, :] = ori
    res[:,2*512:, :] = img
    
    thresh_pred_img = res[:,2*512:, :]
    gt_img = res[:,0:512,:]
    auc_lst.append(calc_auc(thresh_pred_img / 255.0, gt_img / 255.0))
    acc_lst.append(calc_acc(thresh_pred_img / 255.0, gt_img / 255.0))
    sen_lst.append(calc_sen(thresh_pred_img / 255.0, gt_img / 255.0))
    fdr_lst.append(calc_fdr(thresh_pred_img / 255.0, gt_img / 255.0))
    spe_lst.append(calc_spe(thresh_pred_img / 255.0, gt_img / 255.0))
    kappa_lst.append(calc_kappa(thresh_pred_img / 255.0, gt_img / 255.0))
    gmean_lst.append(calc_gmean(thresh_pred_img / 255.0, gt_img / 255.0))
    iou_lst.append(calc_iou(thresh_pred_img / 255.0, gt_img / 255.0))
    dice_lst.append(calc_dice(thresh_pred_img / 255.0, gt_img / 255.0))
  

    print(save_dir + "---" + img_path)
    
    cv2.imwrite(save_dir + img_path, res.astype(int))

    

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

print(" KERNEL SIZE 1,1 ")

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







