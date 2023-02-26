import cv2
import math
import numpy as np
from sklearn import metrics
import torch
from torch import nn
from skimage.metrics import structural_similarity as ssim


def calc_loss(pred, target, metrics, bce_weight=0.5, loss_type="mse"):
    if loss_type == "bce":
        loss = nn.BCEWithLogitsLoss()(pred, target)
    elif loss_type == "mse":
        loss = nn.MSELoss()(pred, target)
    elif loss_type == "rmse":
        mse = nn.MSELoss()(pred, target)
        loss = torch.sqrt(mse)
    elif loss_type == "l1loss":
        loss = nn.L1Loss()(pred, target)

    sim = 0
    for i in range(target.size(0)):
        p, t = (
            pred[i].permute(1, 2, 0).detach().cpu().numpy(),
            target[i].permute(1, 2, 0).detach().cpu().numpy(),
        )
        sim += ssim(p, t, data_range=t.max() - t.min(), multichannel=True)

    metrics["similarity"] = sim
    metrics["loss"] += loss.data.cpu().numpy() * target.size(0)
    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    return "{}: {}".format(phase, ", ".join(outputs))


def max_fusion(x, y):
    assert x.shape == y.shape

    return np.maximum(x, y)


def extract_mask(pred_arr, gt_arr, mask_arr=None):
    # we want to make them into vectors
    pred_vec = pred_arr.flatten()
    gt_vec = gt_arr.flatten()

    if mask_arr is not None:
        mask_vec = mask_arr.flatten()
        idx = list(np.where(mask_vec == 0)[0])

        pred_vec = np.delete(pred_vec, idx)
        gt_vec = np.delete(gt_vec, idx)

    return pred_vec, gt_vec


def calc_auc(pred_arr, gt_arr, mask_arr=None):
    pred_vec, gt_vec = extract_mask(pred_arr, gt_arr, mask_arr=mask_arr)
    roc_auc = metrics.roc_auc_score(gt_vec, pred_vec)

    return roc_auc


def numeric_score(
    pred_arr, gt_arr, kernel_size=(1, 1)
):  # DCC & ROSE-2: kernel_size=(3, 3)
    """Computation of statistical numerical scores:

    * FP = False Positives
    * FN = False Negatives
    * TP = True Positives
    * TN = True Negatives

    return: tuple (FP, FN, TP, TN)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    dilated_gt_arr = cv2.dilate(gt_arr, kernel, iterations=1)

    FP = np.float(np.sum(np.logical_and(pred_arr == 1, dilated_gt_arr == 0)))
    FN = np.float(np.sum(np.logical_and(pred_arr == 0, gt_arr == 1)))
    TP = np.float(np.sum(np.logical_and(pred_arr == 1, dilated_gt_arr == 1)))
    TN = np.float(np.sum(np.logical_and(pred_arr == 0, gt_arr == 0)))

    return FP, FN, TP, TN


def calc_acc(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    acc = (TP + TN) / (FP + FN + TP + TN)

    return acc


def calc_sen(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    sen = TP / (FN + TP + 1e-12)

    return sen


def calc_fdr(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    fdr = FP / (FP + TP + 1e-12)

    return fdr


def calc_spe(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    spe = TN / (FP + TN + 1e-12)

    return spe


def calc_gmean(
    pred_arr, gt_arr, kernel_size=(1, 1)
):  # DCC & ROSE-2: kernel_size=(3, 3)
    sen = calc_sen(pred_arr, gt_arr, kernel_size=kernel_size)
    spe = calc_spe(pred_arr, gt_arr, kernel_size=kernel_size)

    return math.sqrt(sen * spe)


def calc_kappa(
    pred_arr, gt_arr, kernel_size=(1, 1)
):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size=kernel_size)
    matrix = np.array([[TP, FP], [FN, TN]])
    n = np.sum(matrix)

    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col

    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)

    return (po - pe) / (1 - pe)


def calc_iou(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    iou = TP / (FP + FN + TP + 1e-12)

    return iou


def calc_dice(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    dice = 2.0 * TP / (FP + FN + 2.0 * TP + 1e-12)

    return dice
