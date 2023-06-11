import numpy as np
import cv2
import os
from tqdm import tqdm
import re
import torch
from Model import UNet, UNet_multitask, UNet_attention, UNet_fourier1, UNet_fourier1_2
from torchvision import transforms
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import yaml
from skimage import metrics
from sklearn.metrics import roc_auc_score
# scipy
# from sklearn.metrics import auc
from scipy.spatial.distance import directed_hausdorff
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
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.precision = []
        self.recall = []
        self.accuracy = []
        self.f1 = []
        self.mse_list = []
        self.dice_list = []
        self.iou_list = []
        self.hausdorff_distance = []
        
    def mean_square_error(self, y_gt, y_pred):
        number_of_pixel = y_gt.shape[0] * y_gt.shape[1]
        mse = np.sum((y_gt - y_pred)**2)/number_of_pixel
        self.mse_list.append(mse)

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
        
        print('TP IST:',tp_pp)
        # true negatives
        tn_pp = np.sum((y_pred == 0) & (y_gt == 0))
        # false positives
        fp_pp = np.sum((y_pred == 1) & (y_gt_tolerated == 0))
        # false negatives
        fn_pp = np.sum((y_pred == 0) & (y_gt == 1))

        self.tp += tp_pp
        self.fp += fp_pp
        self.tn += tn_pp
        self.fn += fn_pp

        self.precision.append(tp_pp / (tp_pp + fp_pp + smoothening_factor))
        self.recall.append(tp_pp / (tp_pp + fn_pp))
        self.accuracy.append((tp_pp + tn_pp) / (tp_pp + tn_pp + fp_pp + fn_pp))
        self.f1.append(2 * tp_pp / (2 * tp_pp + fp_pp + fn_pp))

        intersection = np.sum((y_pred == 1) * (y_gt_tolerated == 1))
        y_true_area = np.sum((y_gt_tolerated == 1))
        y_pred_area = np.sum((y_pred == 1))
        combined_area = y_true_area + y_pred_area
        iou_score = (intersection + smoothening_factor) / \
            (combined_area - intersection + smoothening_factor)
        dice_score = 2 * ((intersection + smoothening_factor) /
                          (combined_area + smoothening_factor))
        self.dice_list.append(dice_score)
        self.iou_list.append(iou_score)
        if not np.all(y_pred == 0) or np.all(y_pred == 1):
            self.hausdorff_distance.append(
                metrics.hausdorff_distance(y_pred, y_gt_tolerated))
        else:
            print(metrics.hausdorff_distance(y_pred, y_gt_tolerated))
            
        self.mean_square_error(y_gt_tolerated,y_pred)

    

    def calculate_metrics(self, mse=True, g_mean=True, kappa=True, fdr=True, hausdorff_distance=True):
        f = open(os.path.join(self.save_dir, 'result.txt'), 'w')

        # Pixel-wise analysis:
        f.write('Pixel-wise analysis:\n')
        
        tp_total = self.tp
        fn_total = self.fn
        tn_total = self.tn
        fp_total = self.fp
        ry_true = np.concatenate([np.ones(tp_total + fn_total), np.zeros(tn_total + fp_total)])
        ry_pred = np.concatenate([np.ones(tp_total + fp_total), np.zeros(tn_total + fn_total)])
        roc = roc_auc_score(ry_true, ry_pred)
        
        f.write('AUC: {}\n'.format(roc))

        precision = round(self.tp / (self.tp + self.fp), 3)
        recall = round(self.tp / (self.tp + self.fn), 3)
        f1_score = round(2 * precision * recall / (precision + recall), 3)
        acc = round((self.tp + self.tn) /
                    (self.tp + self.tn + self.fp + self.fn), 3)
        dice_score = round(
            (2*self.tp)/(self.fp+self.fn+(2*self.tp)), 3)

        f.write('precision: {}\n'.format(precision))
        f.write('recall: {}\n'.format(recall))
        f.write('f1: {}\n'.format(f1_score))
        f.write("ACC:"+str(acc)+'\n')
        f.write("Dice Score:"+str(dice_score)+'\n')

        # Image-wise analysis:
        f.write('\n')
        f.write('\n')
        f.write('Image-wise analysis:\n')

        precision = round(sum(self.precision)/len(self.precision), 3)
        recall = round(sum(self.recall)/len(self.recall), 3)
        f1_score = round(sum(self.f1)/len(self.f1), 3)
        acc = round(sum(self.accuracy)/len(self.accuracy), 3)
        dice_score = round(sum(self.dice_list)/len(self.dice_list), 3)
        iou_based_image = round(sum(self.iou_list)/len(self.iou_list), 3)

        f.write('precision: {}\n'.format(precision))
        f.write('recall: {}\n'.format(recall))
        f.write('f1: {}\n'.format(f1_score))
        f.write("ACC:"+str(acc)+'\n')
        f.write("Dice Score:"+str(dice_score)+'\n')
        f.write("IOU Score:" + str(iou_based_image)+'\n')

        if mse:
            mean_mse_score = sum(self.mse_list)/len(self.mse_list)
            f.write("Mean MSE:"+str(mean_mse_score)+'\n')
            # Plot iou histogram
            plt.figure(figsize=(10, 8))

            plt.hist(self.mse_list)
            plt.xlabel('individual MSE')
            plt.savefig("{}/mse_histogram.png".format(self.save_dir))
            plt.clf()

        f.write('\n')
        f.write('\n')
        f.write('Additional Metrics:\n')
        if g_mean:
            sensivity = round(self.tp / (self.tp+self.fn + 1e-12), 3)
            specifity = round(self.tn / (self.tn+self.fp + 1e-12), 3)
            g_mean_score = round(np.sqrt(sensivity*specifity), 3)
            f.write("G-mean:"+str(g_mean_score)+'\n')
        if kappa:
            acc = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
            pe = ((self.tp+self.fn)*(self.tp+self.fp) +
                  (self.tn+self.fp)*(self.tn+self.fn))/(self.tp+self.tn+self.fp+self.fn)**2
            kappa_score = round((acc-pe)/(1-pe), 3)
            f.write("Kappa Score:"+str(kappa_score)+'\n')

        if fdr:
            fdr_score = round(self.fp/(self.fp+self.tp + 1e-12), 3)
            f.write("FDR:"+str(fdr_score)+'\n')
        if hausdorff_distance:
            hausdorff_distance_avg = round(
                sum(self.hausdorff_distance)/len(self.hausdorff_distance), 3)
            hausdorff_distance_max = round(max(self.hausdorff_distance), 3)
            f.write("Hausdorff Distance Avg:"+str(hausdorff_distance_avg)+'\n')
            f.write("Hausdorff Distance Max:"+str(hausdorff_distance_max)+'\n')

        f.close()


def pre_process_rgb(img):
    img = np.float32(img)
    # img = (img - img.mean()) / img.std()
    # HW to CHW (for gray scale)
    img[:, :, 0] = (img[:, :, 0] - img[:, :, 0].mean()
                    ) / img[:, :, 0].std()
    img[:, :, 1] = (img[:, :, 1] - img[:, :, 1].mean()
                    ) / img[:, :, 1].std()
    img[:, :, 2] = (img[:, :, 2] - img[:, :, 2].mean()
                    ) / img[:, :, 2].std()

    # HWC to CHW, BGR to RGB (for three channel)
    img = img.transpose((2, 0, 1))[::-1]
    # add batch
    img = np.expand_dims(img, 0)
    img = torch.as_tensor(img.copy())

    return img


def pre_process(img):
    img = np.float32(img)
    img = (img - img.mean()) / img.std()
    # HW to CHW (for gray scale)
    img = np.expand_dims(img, 0)
    # add batch
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


def save_visuals(img_org, mask_img, prediction, save_dir):
    fig, axs = plt.subplots(1, 3)
    fig.set_figheight(12)
    fig.set_figwidth(30)
    if len(img_org.shape) == 3:
        axs[0].imshow(img_org)
        axs[0].title.set_text('image')
    else:
        axs[0].imshow(img_org, cmap='gray')
        axs[0].title.set_text('image')
    axs[1].imshow(mask_img, cmap='gray')
    axs[1].title.set_text('label')
    axs[2].imshow(prediction, cmap='gray')
    axs[2].title.set_text('prediction')
    fig.savefig(save_dir)
    fig.clf()
    plt.close(fig)


def test_single(model, device, input_size, anydepth, image_list, output_save_dir):
    results_save_dir_images = os.path.join(output_save_dir, 'images')
    if not os.path.exists(results_save_dir_images):
        os.mkdir(results_save_dir_images)
    results = Results(output_save_dir, 3)
    ch = 1
    for img_path in tqdm(image_list):
        image_name = img_path.split('/')[-1]
        image_name = image_name[:image_name.rfind('.')]
        
        print('Depths is :',anydepth)

        if anydepth:
            img_org = cv2.resize(cv2.imread(
                img_path, cv2.IMREAD_ANYDEPTH), input_size)
            img = pre_process(img_org)
        else:
            if ch == 3:
                #img_org = cv2.resize(cv2.imread(img_path), input_size)
                img_org = cv2.resize(cv2.imread(img_path), input_size)/255
                img = pre_process_rgb(img_org)
            elif ch == 1:
                img_org = cv2.resize(cv2.imread(img_path, 0), input_size)
                img = pre_process(img_org)
            else:
                raise ValueError('channel must be 1 or 3')

        # read binary mask
        mask_path = img_path[:img_path.rfind('.')] + '_label.png'
        #mask_img = cv2.resize(cv2.imread(mask_path, 0), input_size)*255
        mask_img = cv2.resize(cv2.imread(mask_path, 0), input_size)

        _, gt_binary_mask = cv2.threshold(mask_img, 125, 1, cv2.THRESH_BINARY)

        pred_bin = model(img.to(device))
        pred_bin = torch.sigmoid(pred_bin)
        pred_bin = pred_bin.data.cpu().numpy()
        pred_bin = pred_bin[0, 0]
        pred_bin[pred_bin >= 0.5] = 1
        pred_bin[pred_bin < 0.5] = 0
        pred_bin_img = np.array(pred_bin * 255, np.uint8)

        results.binary_metrics(gt_binary_mask, pred_bin)

        if anydepth:
            img_org_vis = (img_org/256).astype('uint8')
        else:
            img_org_vis = img_org

        save_visuals(img_org_vis, mask_img, pred_bin_img,
                     os.path.join(results_save_dir_images, image_name+'.png')
                     )

        # save_img_dist = np.hstack(
        #     [img_org, seperater, mask_dist, seperater, pred_dist])
        # cv2.imwrite(os.path.join(results_save_dir_images,
        #             image_name+'_dist.png'), save_img_dist)

    results.calculate_metrics(mse=True, g_mean=True,
                              kappa=True, fdr=True, hausdorff_distance=True)


def test_multitask(model, device, input_size, anydepth, image_list, output_save_dir):
    results_save_dir_images = os.path.join(output_save_dir, 'images')
    if not os.path.exists(results_save_dir_images):
        os.mkdir(results_save_dir_images)
    results = Results(output_save_dir, 3)
    ch = model.n_channels
    for img_path in tqdm(image_list):
        image_name = img_path.split('/')[-1]
        image_name = image_name[:image_name.rfind('.')]

        if anydepth:
            img_org = cv2.resize(cv2.imread(
                img_path, cv2.IMREAD_ANYDEPTH), input_size)
            img = pre_process(img_org)
        else:
            if ch == 3:
                img_org = cv2.resize(cv2.imread(img_path), input_size)/255
                img = pre_process_rgb(img_org)
            elif ch == 1:
                img_org = cv2.resize(cv2.imread(img_path, 0), input_size)
                img = pre_process(img_org)
            else:
                raise ValueError('channel must be 1 or 3')

        # # read dist mask
        # mask_path = img_path[:img_path.rfind('.')] + '_dist_label.png'
        # mask_dist = cv2.resize(cv2.imread(mask_path, 0), (512, 512))

        # read binary mask
        mask_path = img_path[:img_path.rfind('.')] + '_label.png'
        mask_img = cv2.resize(cv2.imread(mask_path, 0), input_size)

        _, gt_binary_mask = cv2.threshold(mask_img, 125, 1, cv2.THRESH_BINARY)

        pred_bin, pred_dist = model(img.to(device))
        # pred_dist = post_process_reg(pred_dist)
        pred_bin = post_process_binary(pred_bin)
        pred_bin_img = np.array(pred_bin * 255, np.uint8)

        results.binary_metrics(gt_binary_mask, pred_bin)

        if anydepth:
            img_org_vis = (img_org/256).astype('uint8')
        else:
            img_org_vis = img_org

        save_visuals(img_org_vis, mask_img, pred_bin_img,
                     os.path.join(results_save_dir_images, image_name+'.png')
                     )
    results.calculate_metrics(mse=True, g_mean=True,
                              kappa=True, fdr=True, hausdorff_distance=True)


def main(cfg, model_path):
    # model configs
    # h, w -> (w, h)
    input_size = (cfg['model_config']['input_size'][1],
                  cfg['model_config']['input_size'][0])
    num_class = cfg['model_config']['num_class']
    ch = cfg['model_config']['channel']
    initial_filter_size = cfg['model_config']['initial_filter_size'][0]
    kernel_size = cfg['model_config']['kernel'][0]
    anydepth = cfg['model_config']['anydepth']

    # train configs
    use_cuda = cfg['train_config']['use_cuda']

    # dataset configs
    test_path = cfg['dataset_config']['test_path']
    image_list = get_image_list(test_path)
    output_save_dir = cfg['dataset_config']['save_dir']

    class_names = cfg['dataset_config']['class_names']
    model_type = cfg['model_config']['model_type']
    dropout = cfg['model_config']['dropout']
    dropout_p = float(cfg['model_config']['drop_out_rate'][0])

    if model_type == 'single':
        
        print('dropout is P :',dropout_p)
        
        if dropout:
            model = UNet(ch, num_class, initial_filter_size,use_cuda, dropout, dropout_p)
            
        else:
            model = UNet(ch, num_class, initial_filter_size,use_cuda)
                     
        # model = UNet_BS([1, 32, 64, 128, 256, 512], "parameters", "dropout")

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
        test_single(model, device, input_size, anydepth,
                    image_list, output_save_dir)
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

        test_multitask(model, device, input_size, anydepth,
                    image_list, output_save_dir)

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
    elif model_type == 'fourier1_2':
        model = UNet_fourier1_2(ch, num_class, initial_filter_size, use_cuda)
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

        test_multitask(model, device, input_size, anydepth,
                       image_list, output_save_dir)
    else:
        raise ValueError('Invalid model_type "%s"' % model_type)


if __name__ == "__main__":
    #args = parse_args()
    #config_path = args.config
    model_path = 'expE150W08_multitask_64F-Seed126/models/last_epoch.pt'
    config_path = 'rose_config.yml'
    with open(config_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    main(cfg, model_path)
