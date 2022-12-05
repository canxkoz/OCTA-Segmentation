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

class_names = ['background', 'vein']
image_ext = ['.png', '.jpg']
val_path = '/home/caki/desktop/projects/unet/unet-tf/rose2/test'
input_size = (512,512)
use_cuda = True
device = "cuda:0"
dtype = torch.cuda.FloatTensor
num_class = 1
ch = 1
output_save_dir = 'exp5'
model_path = './exp5/models/epoch49.pt'



results_save_dir_images = os.path.join(output_save_dir, 'images2')
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


def class_wise_metrics(y_true, y_pred, num_of_class):
    class_wise_iou = []
    class_wise_dice_score = []

    smoothening_factor = 0.00001

    for i in range(num_of_class):

        intersection = np.sum((y_pred == i) * (y_true == i))
        y_true_area = np.sum((y_true == i))
        y_pred_area = np.sum((y_pred == i))
        combined_area = y_true_area + y_pred_area

        iou = (intersection + smoothening_factor) / \
            (combined_area - intersection + smoothening_factor)
        class_wise_iou.append(iou)

        dice_score = 2 * ((intersection + smoothening_factor) /
                          (combined_area + smoothening_factor))
        class_wise_dice_score.append(dice_score)

    return class_wise_iou, class_wise_dice_score


def test_single():

    model = UNet(ch, num_class, use_cuda)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)


    image_list = get_image_list(val_path)
    for img_path in tqdm(image_list):
        image_name = img_path.split('/')[-1]
        image_name = image_name[:image_name.rfind('.')]
        simple_transform = transforms.ToTensor()

        # read dist mask
        mask_path = img_path[:img_path.rfind('.')] + '_dist_label.png'
        mask_dist = cv2.resize(cv2.imread(mask_path, 0), (512, 512))

        img_org = cv2.resize(cv2.imread(img_path, 0), (512, 512))
        img = img_org.astype("float32")/255.
        img = simple_transform(img).unsqueeze(0).to(device)
        pred = model(img)

        pred = pred.data.cpu().numpy()
        pred = pred[0,0]
        pred_img = np.array(pred * 255, np.uint8)

        seperater = np.zeros([img_org.shape[1], 15], dtype=np.uint8)
        seperater.fill(155)

        save_img_dist = np.hstack(
            [img_org, seperater, mask_dist, seperater, pred_img])
        cv2.imwrite(os.path.join(results_save_dir_images,
                    image_name+'_dist.png'), save_img_dist)


def test_multitask():

    model = UNet_mt(ch, num_class, use_cuda)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)


    image_list = get_image_list(val_path)
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

        #postprocess binary output
        pred_bin = torch.sigmoid(pred_bin)
        pred_bin = pred_bin.data.cpu().numpy()
        pred_bin = pred_bin[0,0]
        pred_bin_img = np.array(pred_bin * 255, np.uint8)

        #postprocess regression output
        amin = torch.amin(pred_dist, dim=(-2,-1))
        pred_dist = torch.sub(pred_dist, amin.reshape((amin.shape[0], amin.shape[1], 1, 1)))
        pred_dist = pred_dist[0,0]
        pred_dist = pred_dist.detach().cpu().numpy() * 255



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

if __name__ == "__main__":
    test_multitask()
