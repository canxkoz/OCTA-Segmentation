#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 14:31:40 2022

@author: onurcaki
"""
import argparse
import os
import re
import shutil
import argparse
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

image_ext = ['.png']

def natural_sort(l):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            print()
            if '_label' in filename:
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]
                if ext in image_ext:
                    image_names.append(apath)
    return natural_sort(image_names)

def main():
    parser = argparse.ArgumentParser(
        description="Convert binary to distance")
    parser.add_argument('-i', '--input_folder_path',
                        help='file path where binary images are located.', required=True)
    parser.add_argument('-o', '--output_folder_path',
                        help='file path where distances will be saved', required=True)
    args = parser.parse_args()
    
    path = args.input_folder_path
    output_path = args.output_folder_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    image_list = get_image_list(path)
    for img_path in tqdm(image_list):
        
        img = cv2.imread(img_path,0)
        img_invert = (255-img)
        _ , img_invert = cv2.threshold(img_invert, 20, 1, cv2.THRESH_BINARY)
        img_invert_dist = cv2.distanceTransform(img_invert, cv2.DIST_L2, 5)
        img_invert_dist = img_invert_dist.astype(np.uint8)
        
        # import matplotlib.pyplot as plt
        # plt.imshow(img, cmap='gray')
        # plt.imshow(img_invert, cmap='gray')
        # plt.imshow(img_invert_dist, cmap='gray')     
        
        img_name = img_path[:img_path.rfind('_label')] + '_dist_label.png'
        cv2.imwrite(os.path.join(output_path,img_name.split('/')[-1]),img_invert_dist)

if __name__ == "__main__":
    main()