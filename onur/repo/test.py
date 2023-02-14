import numpy as np
import cv2
import os
from tqdm import tqdm
import re
import torch
from Model import UNet, UNet_multitask, UNet_attention, UNet_fourier1
from torchvision import transforms
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import yaml
# from sklearn.metrics import auc

image_ext = ['.png', '.jpg']


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', help='the config path')
    ap.add_argument('model_path', help='model path')

    args = ap.parse_args()
    return args


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





class Results:
    def __init__(self, save_dir, tolerance=0):
        self.save_dir = save_dir
        self.tolerance = tolerance
        self.tp = []
        self.fp = []
        self.tn = []
        self.fn = []
        self.mse_list = []
        self.dice_list = []
        self.iou_list = []
        self.mean_mse_score = 0

        self.auc = 0
        self.acc = 0
        self.g_mean = 0
        self.kappa = 0
        self.dice_score = 0
        self.fdr = 0  # false discovery rate

    def binary_metrics(self, y_gt, y_pred):
        """
        calculate metrics threating each pixel as a sample
        """
        smoothening_factor = 1e-6
        if self.tolerance != 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.tolerance, self.tolerance))
            y_gt_tolerated = cv2.dilate(y_gt, kernel, iterations=1)
        else:
            y_gt_tolerated = y_gt
        # true positives
        tp_pp = np.sum((y_pred == 1) & (y_gt_tolerated == 1))
        # true negatives
        tn_pp = np.sum((y_pred == 0) & (y_gt == 0))
        # false positives
        fp_pp = np.sum((y_pred == 1) & (y_gt_tolerated == 0))
        # false negatives
        fn_pp = np.sum((y_pred == 0) & (y_gt == 1))

        self.tp.append(tp_pp)
        self.fp.append(fp_pp)
        self.tn.append(tn_pp)
        self.fn.append(fn_pp)

        intersection = np.sum((y_pred == 1) * (y_gt_tolerated == 1))
        y_true_area = np.sum((y_gt_tolerated == 1))
        y_pred_area = np.sum((y_pred == 1))
        combined_area = y_true_area + y_pred_area
        iou_score = (intersection + smoothening_factor) / \
            (combined_area - intersection + smoothening_factor)
        dice_score = 2 * ((intersection) /
                          (combined_area + smoothening_factor))
        self.dice_list.append(dice_score)
        self.iou_list.append(iou_score)

    def mean_square_error(self, y_gt, y_pred):
        number_of_pixel = y_gt.shape[0] * y_gt.shape[1]
        mse = np.sum((y_gt - y_pred)**2)/number_of_pixel
        self.mse_list.append(mse)

    def calculate_metrics(self, mse=True, auc=True, acc=True, g_mean=True, kappa=True, dice_score=True, iou_score=True, fdr=True):
        f = open(os.path.join(self.save_dir, 'result.txt'), 'w')
        tp = sum(self.tp) / len(self.tp)
        fp = sum(self.fp) / len(self.fp)
        tn = sum(self.tn) / len(self.tn)
        fn = sum(self.fn) / len(self.fn)
        precision = round(tp / (tp + fp), 3)
        recall = round(tp / (tp + fn), 3)
        f1_score = round(2 * precision * recall / (precision + recall), 3)
        f.write('precision: {}\n'.format(precision))
        f.write('recall: {}\n'.format(recall))
        f.write('f1: {}\n'.format(f1_score))

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
            self.acc = round((tp + tn) /
                             (tp + tn + fp + fn), 3)
            f.write("ACC:"+str(self.acc)+'\n')
        if g_mean:
            sensivity = round(tp / (tp+fn + 1e-12), 3)
            specifity = round(tn / (tn+fp + 1e-12), 3)
            self.g_mean = round(np.sqrt(sensivity*specifity), 3)
            f.write("G-mean:"+str(self.g_mean)+'\n')
        if kappa:
            pe = ((tp+fn)*(tp+fp) +
                  (tn+fp)*(tn+fn))/(tp+tn+fp+fn)**2
            self.kappa = round((self.acc-pe)/(1-pe), 3)
            f.write("Kappa Score:"+str(self.kappa)+'\n')
        if dice_score:
            self.dice_score = round(
                (2*tp)/(fp+fn+(2*tp)), 3)
            f.write("Dice Score:"+str(self.dice_score)+'\n')
            dice_based_image = round(
                sum(self.dice_list)/len(self.dice_list), 3)
            f.write("Dice Score based on individual images:" +
                    str(dice_based_image)+'\n')
        if iou_score:
            iou_based_image = round(sum(self.iou_list)/len(self.iou_list), 3)
            f.write("IOU Score based on individual images:" +
                    str(iou_based_image)+'\n')
        if fdr:
            self.fdr = round(fp/(fp+tp + 1e-12), 3)
            f.write("FDR:"+str(self.fdr)+'\n')

        f.close()


def pre_process(img):
    img = np.float32(img)
    img = (img - img.mean()) / img.std()
    # HW to CHW (for gray scale)
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 0)

    # HWC to CHW, BGR to RGB (for three channel)
    # img = img.transpose((2, 0, 1))[::-1]
    img = torch.as_tensor(img)
    return img


def post_process_binary(pred_bin):
    pred_bin = torch.sigmoid(pred_bin)
    pred_bin = pred_bin.data.cpu().numpy()
    pred_bin = pred_bin[0, 0]
    pred_bin[pred_bin >= 0.5] = 1
    pred_bin[pred_bin < 0.5] = 0
    return pred_bin


def post_process_reg(pred_dist_map):
    amin = torch.amin(pred_dist_map, dim=(-2, -1))
    pred_dist_map = torch.sub(pred_dist_map, amin.reshape(
        (amin.shape[0], amin.shape[1], 1, 1)))
    pred_dist_map = pred_dist_map[0].squeeze()
    pred_dist_map = pred_dist_map.detach().cpu().numpy() * 255
    return pred_dist_map


def test_single(model, device, input_size, image_list, output_save_dir):
    results_save_dir_images = os.path.join(output_save_dir, 'images')
    if not os.path.exists(results_save_dir_images):
        os.mkdir(results_save_dir_images)
    results = Results(output_save_dir, 0)
    for img_path in tqdm(image_list):
        image_name = img_path.split('/')[-1]
        image_name = image_name[:image_name.rfind('.')]

        # read binary mask
        mask_path = img_path[:img_path.rfind('.')] + '_label.png'
        mask = cv2.resize(cv2.imread(mask_path, 0), (512, 512))*255

        _, gt_binary_mask = cv2.threshold(mask, 125, 1, cv2.THRESH_BINARY)

        img_org = cv2.resize(cv2.imread(
            img_path, cv2.IMREAD_ANYDEPTH), input_size)

        img = pre_process(img_org)
        pred_bin = model(img.to(device))
        pred_bin = torch.sigmoid(pred_bin)
        pred_bin = pred_bin.data.cpu().numpy()
        pred_bin = pred_bin[0, 0]
        pred_bin[pred_bin >= 0.5] = 1
        pred_bin[pred_bin < 0.5] = 0
        pred_bin_img = np.array(pred_bin * 255, np.uint8)

        results.binary_metrics(gt_binary_mask, pred_bin)

        img8_8bit = (img_org/256).astype('uint8')

        seperater = np.zeros([img_org.shape[1], 15], dtype=np.uint8)
        seperater.fill(155)

        # save_img_dist = np.hstack(
        #     [img_org, seperater, mask_dist, seperater, pred_dist])
        # cv2.imwrite(os.path.join(results_save_dir_images,
        #             image_name+'_dist.png'), save_img_dist)

        save_img_bin = np.hstack(
            [img8_8bit, seperater, mask, seperater, pred_bin_img])
        cv2.imwrite(os.path.join(results_save_dir_images,
                    image_name+'.png'), save_img_bin)
    results.calculate_metrics(mse=False, acc=True)


def test_multitask(model, device, input_size, image_list, output_save_dir):
    results_save_dir_images = os.path.join(output_save_dir, 'images')
    if not os.path.exists(results_save_dir_images):
        os.mkdir(results_save_dir_images)
    results = Results(output_save_dir, 1)
    for img_path in tqdm(image_list):
        image_name = img_path.split('/')[-1]
        image_name = image_name[:image_name.rfind('.')]

        # # read dist mask
        # mask_path = img_path[:img_path.rfind('.')] + '_dist_label.png'
        # mask_dist = cv2.resize(cv2.imread(mask_path, 0), (512, 512))

        # read binary mask
        mask_path = img_path[:img_path.rfind('.')] + '_label.png'
        mask = cv2.resize(cv2.imread(mask_path, 0), (512, 512))*255

        _, gt_binary_mask = cv2.threshold(mask, 125, 1, cv2.THRESH_BINARY)

        img_org = cv2.resize(cv2.imread(
            img_path, cv2.IMREAD_ANYDEPTH), input_size)

        img = pre_process(img_org)
        pred_bin, pred_dist = model(img.to(device))
        #pred_dist = post_process_reg(pred_dist)
        pred_bin = post_process_binary(pred_bin)
        pred_bin_img = np.array(pred_bin * 255, np.uint8)

        results.binary_metrics(gt_binary_mask, pred_bin)

        img8_8bit = (img_org/256).astype('uint8')

        seperater = np.zeros([img_org.shape[1], 15], dtype=np.uint8)
        seperater.fill(155)

        # save_img_dist = np.hstack(
        #     [img8_8bit, seperater, mask_dist, seperater, pred_dist])
        # cv2.imwrite(os.path.join(results_save_dir_images,
        #             image_name+'_dist.png'), save_img_dist)

        save_img_bin = np.hstack(
            [img8_8bit, seperater, mask, seperater, pred_bin_img])
        cv2.imwrite(os.path.join(results_save_dir_images,
                    image_name+'.png'), save_img_bin)
    results.calculate_metrics(mse=False, acc=True)


def main(cfg, model_path):
    # model configs
    input_size = (cfg['model_config']['input_size'][0],
                  cfg['model_config']['input_size'][1])
    num_class = cfg['model_config']['num_class']
    ch = cfg['model_config']['channel']
    initial_filter_size = cfg['model_config']['initial_filter_size'][0]
    kernel_size = cfg['model_config']['kernel'][0]

    # train configs
    use_cuda = cfg['train_config']['use_cuda']

    # dataset configs
    test_path = cfg['dataset_config']['test_path']
    image_list = get_image_list(test_path)
    output_save_dir = cfg['dataset_config']['save_dir']

    class_names = cfg['dataset_config']['class_names']
    model_type = cfg['model_config']['model_type']

    if model_type == 'single':
        model = UNet(ch, num_class, initial_filter_size, use_cuda)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        if use_cuda:
            print('Gpu available')
            print(torch.cuda.get_device_name(0))
            device = "cuda:0"
            dtype = torch.cuda.FloatTensor
            model.to(device=device)
        else:
            model.to(device="cpu")
        test_single(model, device, input_size, image_list, output_save_dir)
    elif model_type == 'multi_task':
        model = UNet_multitask(ch, num_class, initial_filter_size, use_cuda)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        if use_cuda:
            print('Gpu available')
            print(torch.cuda.get_device_name(0))
            device = "cuda:0"
            dtype = torch.cuda.FloatTensor
            model.to(device=device)
        else:
            model.to(device="cpu")

        test_multitask(model, device, input_size, image_list, output_save_dir)

    elif model_type == 'attention':
        model = UNet_attention(
            ch, num_class, initial_filter_size, use_cuda)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        if use_cuda:
            print('Gpu available')
            print(torch.cuda.get_device_name(0))
            device = "cuda:0"
            dtype = torch.cuda.FloatTensor
            model.to(device=device)
        else:
            model.to(device="cpu")
        test_single(model, device, input_size, image_list, output_save_dir)
    elif model_type == 'fourier1':
        model = UNet_fourier1(ch, num_class, initial_filter_size, use_cuda)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        if use_cuda:
            print('Gpu available')
            print(torch.cuda.get_device_name(0))
            device = "cuda:0"
            dtype = torch.cuda.FloatTensor
            model.to(device=device)
        else:
            model.to(device="cpu")

        test_multitask(model, device, input_size, image_list, output_save_dir)
    else:
        raise ValueError('Invalid model_type "%s"' % model_type)


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    model_path = args.model_path
    # config_path = 'config.yml'
    with open(config_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    main(cfg, model_path)
