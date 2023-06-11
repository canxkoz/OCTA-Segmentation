import torch
from torch.utils.data import Dataset
import os
import re
from torchvision import transforms
import numpy as np
import cv2
import random
import torchvision.transforms.functional as TF

image_ext = ['.jpg', '.jpeg', '.webp', '.bmp', '.png', '.tif']


class Data_Reg_Binary(Dataset):
    def __init__(self, data_path, ch=1, input_size=(512, 512), augmentation=False):
        super(Data_Reg_Binary, self).__init__()
        self.image_list = self.get_image_list(data_path)
        self.channel = ch
        self.augmentation = augmentation
        self.height = input_size[0]
        self.width = input_size[1]

    def transform_mask(self, img, mask):

        # # Random horizontal flipping
        # if random.random() > 0.5 and self.augmentation:
        #     image = TF.hflip(image)
        #     mask = TF.hflip(mask)

        # # Random vertical flipping
        # if random.random() > 0.5 and self.augmentation:
        #     image = TF.vflip(image)
        #     mask = TF.vflip(mask)

        # # Random rotation
        # if random.random() and self.augmentation > 0.5:
        #     angle = random.randint(10, 350)
        #     image = TF.rotate(image, angle)
        #     mask = TF.rotate(mask, angle)

        # # Brightness
        # if random.random() and self.augmentation > 0.5:
        #     image = TF.adjust_brightness(image, random.uniform(0.5, 1.0))

        # # Contrast
        # if random.random() and self.augmentation > 0.5:
        #     image = TF.adjust_contrast(image, random.uniform(0.5, 1.5))

        # # Gamma
        # if random.random() > 0.5 and self.augmentation:
        #     image = TF.adjust_gamma(image, random.uniform(0.5, 1))

        # # Gaussian Blur
        # if random.random() > 0.5 and self.augmentation:
        #     image = TF.gaussian_blur(image, (3, 3))

        # Normalized
        img = (img - img.mean()) / img.std()
        # HW to CHW (for gray scale)
        img = np.expand_dims(img, 0)
        # HWC to CHW, BGR to RGB (for three channel)
        # img = img.transpose((2, 0, 1))[::-1]
        img = torch.as_tensor(img)

        # for 0 - 255
        # convert tensor with normalizzation
        # gt_mask_bin = TF.to_tensor(gt_mask_bin)

        # for 0 - 1 -2
        mask = mask/255
        mask = np.expand_dims(mask, 0)
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)

        return img, mask

    def __getitem__(self, index):
        

        # read image
        imgPath = self.image_list[index]
        img = cv2.imread(imgPath, cv2.IMREAD_ANYDEPTH)
        r = max(self.width, self.height) / max(img.shape[:2])  # ratio
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            img = cv2.resize(img, (self.width, self.height),
                             interpolation=interp)

        # read target label mask
        gt_mask_path = imgPath[:imgPath.rfind('.')] + '_label.png'
        gt_mask_bin = cv2.imread(gt_mask_path, 0)
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            gt_mask_bin = cv2.resize(gt_mask_bin, (self.width, self.height),
                                     interpolation=interp)

        # Preprocess
        img, gt_mask_bin = self.transform_mask(img, gt_mask_bin)

        # read distance map
        gtPath_dist = imgPath[:imgPath.rfind('.')] + '_dist_label.png'
        gt_dist = cv2.imread(gtPath_dist, 0)
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            gt_dist = cv2.resize(gt_dist, (self.width, self.height),
                                 interpolation=interp)

        # preprocess
        gt_dist = TF.to_tensor(gt_dist)

        return img, gt_mask_bin, gt_dist

    def __len__(self):
        return len(self.image_list)

    def natural_sort(self, l):
        def convert(text): return int(text) if text.isdigit() else text.lower()
        def alphanum_key(key): return [convert(c)
                                       for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    def get_image_list(self, path):
        image_paths = []
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                if '_label' in filename:
                    continue
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]
                if ext in image_ext:
                    image_paths.append(apath)
        return self.natural_sort(image_paths)


class Data_Binary(Dataset):
    def __init__(self, data_path, ch=1, input_size=(512, 512), augmentation=False):
        super(Data_Binary, self).__init__()
        self.image_list = self.get_image_list(data_path)
        self.channel = ch
        self.augmentation = augmentation
        self.height = input_size[0]
        self.width = input_size[1]

    def transform_mask(self, img, mask):

        # # Random horizontal flipping
        # if random.random() > 0.5 and self.augmentation:
        #     image = TF.hflip(image)
        #     mask = TF.hflip(mask)

        # # Random vertical flipping
        # if random.random() > 0.5 and self.augmentation:
        #     image = TF.vflip(image)
        #     mask = TF.vflip(mask)

        # # Random rotation
        # if random.random() and self.augmentation > 0.5:
        #     angle = random.randint(10, 350)
        #     image = TF.rotate(image, angle)
        #     mask = TF.rotate(mask, angle)

        # # Brightness
        # if random.random() and self.augmentation > 0.5:
        #     image = TF.adjust_brightness(image, random.uniform(0.5, 1.0))

        # # Contrast
        # if random.random() and self.augmentation > 0.5:
        #     image = TF.adjust_contrast(image, random.uniform(0.5, 1.5))

        # # Gamma
        # if random.random() > 0.5 and self.augmentation:
        #     image = TF.adjust_gamma(image, random.uniform(0.5, 1))

        # # Gaussian Blur
        # if random.random() > 0.5 and self.augmentation:
        #     image = TF.gaussian_blur(image, (3, 3))

        # Normalized
        img = (img - img.mean()) / img.std()
        # HW to CHW (for gray scale)
        img = np.expand_dims(img, 0)
        # HWC to CHW, BGR to RGB (for three channel)
        # img = img.transpose((2, 0, 1))[::-1]
        img = torch.as_tensor(img)

        # for 0 - 255
        # convert tensor with normalizzation
        # gt_mask_bin = TF.to_tensor(gt_mask_bin)

        # for 0 - 1 -2
        mask = mask/255
        mask = np.expand_dims(mask, 0)
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)

        return img, mask

    def __getitem__(self, index):

        # read image
        imgPath = self.image_list[index]
        img = cv2.imread(imgPath, cv2.IMREAD_ANYDEPTH)
        r = max(self.width,self.height) / max(img.shape[:2])  # ratio
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            img = cv2.resize(img, (self.width, self.height),
                             interpolation=interp)

        # read target label mask
        gt_mask_path = imgPath[:imgPath.rfind('.')] + '_label.png'
        gt_mask_bin = cv2.imread(gt_mask_path, 0)
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            gt_mask_bin = cv2.resize(gt_mask_bin, (self.width, self.height),
                                     interpolation=interp)

        # Preprocess
        img, gt_mask_bin = self.transform_mask(img, gt_mask_bin)

        return img, gt_mask_bin

    def __len__(self):
        return len(self.image_list)

    def natural_sort(self, l):
        def convert(text): return int(text) if text.isdigit() else text.lower()
        def alphanum_key(key): return [convert(c)
                                       for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    def get_image_list(self, path):
        image_paths = []
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                if '_label' in filename:
                    continue
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]
                if ext in image_ext:
                    image_paths.append(apath)
        return self.natural_sort(image_paths)


class Data_Reg_Fourier1(Dataset):
    def __init__(self, data_path, ch=1, input_size=(512, 512), augmentation=False):
        super(Data_Reg_Fourier1, self).__init__()
        self.image_list = self.get_image_list(data_path)
        self.channel = ch
        self.augmentation = augmentation
        self.height = input_size[0]
        self.width = input_size[1]

    def transform_mask(self, img, mask, fmap):

        # # Random horizontal flipping
        # if random.random() > 0.5 and self.augmentation:
        #     image = TF.hflip(image)
        #     mask = TF.hflip(mask)

        # # Random vertical flipping
        # if random.random() > 0.5 and self.augmentation:
        #     image = TF.vflip(image)
        #     mask = TF.vflip(mask)

        # # Random rotation
        # if random.random() and self.augmentation > 0.5:
        #     angle = random.randint(10, 350)
        #     image = TF.rotate(image, angle)
        #     mask = TF.rotate(mask, angle)

        # # Brightness
        # if random.random() and self.augmentation > 0.5:
        #     image = TF.adjust_brightness(image, random.uniform(0.5, 1.0))

        # # Contrast
        # if random.random() and self.augmentation > 0.5:
        #     image = TF.adjust_contrast(image, random.uniform(0.5, 1.5))

        # # Gamma
        # if random.random() > 0.5 and self.augmentation:
        #     image = TF.adjust_gamma(image, random.uniform(0.5, 1))

        # # Gaussian Blur
        # if random.random() > 0.5 and self.augmentation:
        #     image = TF.gaussian_blur(image, (3, 3))

        # Normalized
        img = (img - img.mean()) / img.std()
        # HW to CHW (for gray scale)
        img = np.expand_dims(img, 0)
        # HWC to CHW, BGR to RGB (for three channel)
        # img = img.transpose((2, 0, 1))[::-1]
        img = torch.as_tensor(img)

        # Normalized
        fmap = (fmap - fmap.mean()) / fmap.std()
        # HW to CHW (for gray scale)
        fmap = np.expand_dims(fmap, 0)
        # HWC to CHW, BGR to RGB (for three channel)
        # img = img.transpose((2, 0, 1))[::-1]
        fmap = torch.as_tensor(fmap)


        # for 0 - 255
        # convert tensor with normalizzation
        # gt_mask_bin = TF.to_tensor(gt_mask_bin)

        # for 0 - 1 -2
        mask = mask/255
        mask = np.expand_dims(mask, 0)
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)

        return img, mask, fmap

    def __getitem__(self, index):

        # read image
        imgPath = self.image_list[index]
        img = cv2.imread(imgPath, cv2.IMREAD_ANYDEPTH)
        r = max(self.width, self.height) / max(img.shape[:2])  # ratio
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            img = cv2.resize(img, (self.width, self.height),
                             interpolation=interp)

        # read target label mask
        gt_mask_path = imgPath[:imgPath.rfind('.')] + '_label.png'
        gt_mask_bin = cv2.imread(gt_mask_path, 0)
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            gt_mask_bin = cv2.resize(gt_mask_bin, (self.width, self.height),
                                     interpolation=interp)

        # read distance map
        gtPath_fmap1 = imgPath[:imgPath.rfind('.')] + '_center2.fdmap1'
        gt_fmap1 = np.loadtxt(gtPath_fmap1)
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            gt_dist = cv2.resize(gt_dist, (self.width, self.height),
                                 interpolation=interp)

        # Preprocess
        img, gt_mask_bin, gt_fmap1 = self.transform_mask(
            img, gt_mask_bin, gt_fmap1)

        return img, gt_mask_bin, gt_fmap1

    def __len__(self):
        return len(self.image_list)

    def natural_sort(self, l):
        def convert(text): return int(text) if text.isdigit() else text.lower()
        def alphanum_key(key): return [convert(c)
                                       for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    def get_image_list(self, path):
        image_paths = []
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                if '_label' in filename:
                    continue
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]
                if ext in image_ext:
                    image_paths.append(apath)
        return self.natural_sort(image_paths)
