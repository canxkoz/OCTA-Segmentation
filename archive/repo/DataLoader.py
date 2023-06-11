import torch
from torch.utils.data import Dataset
import os 
import re
from PIL import Image, ImageOps
from torchvision import transforms
import numpy as np
import cv2
image_ext = ['.jpg', '.jpeg', '.webp', '.bmp', '.png']

class Data_Reg_Binary(Dataset):
    def __init__(self, data_path, ch=1, scale_size=(512, 512), augmentation = False):
        super(Data_Reg_Binary, self).__init__()
        self.image_list = self.get_image_list(data_path)
        self.channel = ch
        self.augmentation = augmentation
        self.scale_size = scale_size
        assert self.channel == 1 or self.channel == 3, "the channel must be 1 or 3" # check the channel is 1 or 3
    
    def __getitem__(self, index):

        simple_transform = transforms.ToTensor()

        #read image 
        imgPath = self.image_list[index]
        img = Image.open(imgPath)
        #preprocess
        img = simple_transform(img)

        #read distance map
        gtPath_dist =  imgPath[:imgPath.rfind('.')] + '_dist_label.png'
        gt_dist = Image.open(gtPath_dist)
        #preprocess
        gt_dist = ImageOps.equalize(gt_dist)
        gt_dist = simple_transform(gt_dist)

        #read binary mask
        gt_mask_path = imgPath[:imgPath.rfind('.')] + '_label.png'
        gt_mask_bin = Image.open(gt_mask_path)
        #preprocess
        gt_mask_bin = simple_transform(gt_mask_bin)


        return img, gt_mask_bin, gt_dist

    def __len__(self):
        return len(self.image_list)
    
    def natural_sort(self, l): 
        convert = lambda text: int(text) if text.isdigit() else text.lower() 
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(l, key = alphanum_key)

    def get_image_list(self, path):
        image_paths = []
        global IMAGE_EXT
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                if '_label' in filename:
                    continue
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]
                if ext in image_ext:
                    image_paths.append(apath)
        return self.natural_sort(image_paths)
