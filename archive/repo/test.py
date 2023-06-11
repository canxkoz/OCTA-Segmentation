import numpy as np
import cv2
import os
from tqdm import tqdm
import re
import torch
from Model import UNet, UNet_mt
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
#from sklearn.metrics import auc

class_names = ['background', 'vein']
image_ext = ['.png', '.jpg']
val_path = '/home/caki/desktop/projects/unet/processed_datasets/rose2/test'

input_size = (512,512)
use_cuda = True
device = "cuda:0"
dtype = torch.cuda.FloatTensor
num_class = 1
ch = 1
output_save_dir = 'exp7'
model_path = 'exp7/models/epoch129.pt'
Multitask = True

binary_from_reg = True

results_save_dir_images = os.path.join(output_save_dir, 'images')
if not os.path.exists(results_save_dir_images):
    os.mkdir(results_save_dir_images)


def natural_sort(l):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            if '_label' not in filename:
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]
                if ext in image_ext:
                    image_names.append(apath)
    return natural_sort(image_names)


def post_process_reg(prediction):
    pred_dist_map = prediction[0].squeeze()
    pred_dist_map = pred_dist_map*255
    pred_dist_map = pred_dist_map.astype(np.uint8)
    return pred_dist_map


def post_process_binarymask(prediction):
    pred_mask = prediction[0].squeeze()
    pred_mask = pred_mask*(255/pred_mask.max())
    _, pred_mask = cv2.threshold(pred_mask, 50, 1, cv2.THRESH_BINARY)
    return pred_mask

class Results:
    def __init__(self, save_dir, tolerance = 0):
        self.save_dir = save_dir
        self.tolerance = tolerance
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.mse_list = []
        self.mean_mse_score = 0

        self.auc = 0
        self.acc = 0
        self.g_mean = 0
        self.kappa = 0
        self.dice_score = 0
        self.fdr = 0 #false discovery rate

    def binary_metrics(self, y_gt, y_pred):
        """
        calculate metrics threating each pixel as a sample
        """
        if self.tolerance !=0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.tolerance,self.tolerance))
            y_gt_tolerated = cv2.dilate(y_gt, kernel, iterations=1)
        else:
            y_gt_tolerated = y_gt
        # true positives
        tp_pp = np.sum((y_pred==1) & (y_gt_tolerated==1))
        # true negatives
        tn_pp = np.sum((y_pred==0) & (y_gt==0))
        # false positives
        fp_pp = np.sum((y_pred==1) & (y_gt_tolerated==0))
        # false negatives
        fn_pp = np.sum((y_pred==0) & (y_gt==1))
        
        self.tp += tp_pp
        self.fp += fp_pp
        self.tn += tn_pp
        self.fn += fn_pp

    def mean_square_error(self, y_gt, y_pred):
        number_of_pixel = y_gt.shape[0] * y_gt.shape[1]
        mse = np.sum((y_gt - y_pred)**2)/number_of_pixel
        self.mse_list.append(mse)

    def calculate_metrics(self, mse = True, auc = True, acc = True, g_mean = True, kappa = True, dice_score = True, fdr = True):
        f=open(os.path.join(self.save_dir,'result.txt'),'w')
        if mse:
            self.mean_mse_score = sum(self.mse_list)/len(self.mse_list)
            f.write("Mean MSE:"+str(self.mean_mse_score)+'\n')
            # Plot iou histogram
            plt.figure(figsize=(10, 8))

            plt.hist(self.mse_list)
            plt.xlabel('individual MSE')
            plt.savefig("{}/mse_histogram.png".format(self.save_dir))
            plt.clf()
        if acc:
            self.acc = round((self.tp + self.tn) / (self.tp + self.tn +self.fp + self.fn),3)
            f.write("ACC:"+str(self.acc)+'\n')
        if g_mean:
            sensivity = round(self.tp / (self.tp+self.fn + 1e-12),3)
            specifity = round(self.tn / (self.tn+self.fp + 1e-12),3)
            self.g_mean = round(np.sqrt(sensivity*specifity),3)
            f.write("G-mean:"+str(self.g_mean)+'\n')
        if kappa:
            pe = ((self.tp+self.fn)*(self.tp+self.fp)+
              (self.tn+self.fp)*(self.tn+self.fn))/(self.tp+self.tn+self.fp+self.fn)**2
            self.kappa = round((self.acc-pe)/(1-pe),3)
            f.write("Kappa Score:"+str(self.kappa)+'\n')
        if dice_score:
            self.dice_score = round((2*self.tp)/(self.fp+self.fn+(2*self.tp)),3)
            f.write("Dice Score:"+str(self.dice_score)+'\n')
        if fdr:
            self.fdr = round(self.fp/(self.fp+self.tp+ 1e-12),3)
            f.write("FDR:"+str(self.fdr)+'\n')
            
        f.close()

def test_single():




    model = UNet(ch, num_class, use_cuda)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    image_list = get_image_list(val_path)

    results = Results(output_save_dir,3)
    for img_path in tqdm(image_list):
        image_name = img_path.split('/')[-1]
        image_name = image_name[:image_name.rfind('.')]
        simple_transform = transforms.ToTensor()

        # read binary mask
        mask_path = img_path[:img_path.rfind('.')] + '_label.png'
        mask = cv2.resize(cv2.imread(mask_path, 0), (512, 512))
        _, gt_binary_mask = cv2.threshold(mask, 20, 1, cv2.THRESH_BINARY)

        img_org = Image.open(img_path)
        img = simple_transform(img_org).unsqueeze(0).to(device)
        pred_bin = model(img)

        #postprocess binary output
        pred_bin = torch.sigmoid(pred_bin)
        pred_bin = pred_bin.data.cpu().numpy()
        pred_bin = pred_bin[0,0]
        pred_bin_img = np.array(pred_bin * 255, np.uint8)
        pred_bin[pred_bin>=0.2] = 1
        pred_bin[pred_bin<0.2] = 0

        results.binary_metrics(gt_binary_mask, pred_bin)

        img_org = np.asarray(img_org)

        seperater = np.zeros([img_org.shape[1], 15], dtype=np.uint8)
        seperater.fill(155)

        save_img_bin = np.hstack(
            [img_org, seperater, gt_binary_mask*255, seperater, pred_bin_img])
        cv2.imwrite(os.path.join(results_save_dir_images,
                    image_name+'.png'), save_img_bin)
    results.calculate_metrics(mse=False)





def test_multitask():

    model = UNet_mt(ch, num_class, use_cuda)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    image_list = get_image_list(val_path)

    results = Results(output_save_dir,3)
    for img_path in tqdm(image_list):
        image_name = img_path.split('/')[-1]
        image_name = image_name[:image_name.rfind('.')]
        simple_transform = transforms.ToTensor()

        # read dist mask
        mask_path = img_path[:img_path.rfind('.')] + '_dist_label.png'
        mask_dist = cv2.resize(cv2.imread(mask_path, 0), (512, 512))

        # read binary mask
        mask_path = img_path[:img_path.rfind('.')] + '_label.png'
        mask = cv2.resize(cv2.imread(mask_path, 0), (512, 512))
        _, gt_binary_mask = cv2.threshold(mask, 20, 1, cv2.THRESH_BINARY)

        img_org = Image.open(img_path)
        img = simple_transform(img_org).unsqueeze(0).to(device)
        pred_bin, pred_dist= model(img)

        #postprocess regression output
        amin = torch.amin(pred_dist, dim=(-2,-1))
        pred_dist = torch.sub(pred_dist, amin.reshape((amin.shape[0], amin.shape[1], 1, 1)))
        pred_dist = pred_dist[0,0]
        pred_dist = pred_dist.detach().cpu().numpy() * 255
        results.mean_square_error(mask_dist, pred_dist)

        #postprocess binary output
        if binary_from_reg:
            pred_bin = pred_dist.copy()
            pred_bin[pred_bin>=50] = 1
            pred_bin[pred_bin<50] = 0
            pred_bin_img = np.array(pred_bin, np.uint8)

        else:
            pred_bin = torch.sigmoid(pred_bin)
            pred_bin = pred_bin.data.cpu().numpy()
            pred_bin = pred_bin[0,0]
            pred_bin_img = np.array(pred_bin * 255, np.uint8)
            pred_bin[pred_bin>=0.2] = 1
            pred_bin[pred_bin<0.2] = 0

        results.binary_metrics(gt_binary_mask, pred_bin)

        img_org = np.asarray(img_org)

        seperater = np.zeros([img_org.shape[1], 15], dtype=np.uint8)
        seperater.fill(155)

        save_img_dist = np.hstack(
            [img_org, seperater, mask_dist, seperater, pred_dist])
        cv2.imwrite(os.path.join(results_save_dir_images,
                    image_name+'_dist.png'), save_img_dist)


        save_img_bin = np.hstack(
            [img_org, seperater, gt_binary_mask*255, seperater, pred_bin_img])
        cv2.imwrite(os.path.join(results_save_dir_images,
                    image_name+'.png'), save_img_bin)
    results.calculate_metrics()

if __name__ == "__main__":
    if Multitask:
        test_multitask()
    else:
        test_single()
