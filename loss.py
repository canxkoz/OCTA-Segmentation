import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, multiclass=False):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.multiclass = multiclass

    def dice_score(self, inputs, targets):
        inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_score = (2.*intersection + self.smooth) / \
            (inputs.sum() + targets.sum() + self.smooth)
        return dice_score

    def dice_score_mc(self, inputs, targets):
        inputs = F.softmax(inputs)
        targets = F.one_hot(
            targets.long(), 5).permute(0, 3, 1, 2)
        inputs = inputs.flatten(0, 1)
        targets = targets.flatten(0, 1)
        intersection = (inputs * targets).sum()
        dice_score = (2.*intersection + self.smooth) / \
            (inputs.sum() + targets.sum() + self.smooth)
        return dice_score

    def forward(self, inputs, targets):
        if self.multiclass:
            dice_score = self.dice_score_mc(inputs, targets)
        else:
            dice_score = self.dice_score(inputs, targets)
        return 1 - dice_score


class MultitaskUncertaintyLoss(nn.Module):
    def __init__(self):
        super(MultitaskUncertaintyLoss, self).__init__()

    def forward(self, loss_values, log_var_tasks):
        total_loss = 0
        loss_cls, loss_reg = loss_values
        log_var_task1, log_var_task2 = log_var_tasks
        total_loss += (loss_cls.cpu() / (2*torch.exp(2 * log_var_task1))) + log_var_task1
        total_loss += (loss_reg.cpu() / torch.exp(2 * log_var_task2)) + log_var_task2
        return total_loss


def calc_loss(pred, target, class_weights=None, bce_weight=0.5, loss_type='mse'):

    if loss_type == 'BCE':
        if class_weights is not None:
            # apply class weights to binary cross-entropy loss
            loss = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])(pred, target)
        else:
            loss = nn.BCEWithLogitsLoss()(pred, target)


    if loss_type == 'ce':
        if class_weights is not None:
            # apply class weights to binary cross-entropy loss
            loss = loss = nn.CrossEntropyLoss(weight=class_weights)(pred, target)
        else:
           loss = nn.CrossEntropyLoss()(pred, target)
            
    if loss_type == 'mse':
        loss = nn.MSELoss()(pred, target)
    if loss_type == 'rmse':
        mse = nn.MSELoss()(pred, target)
        loss = torch.sqrt(mse)
    if loss_type == 'l1loss':
        loss = nn.L1Loss()(pred, target)
    if loss_type == 'dice':
        loss = DiceLoss()(pred, target)
    if loss_type == 'dice_bce':
        loss = DiceLoss()(pred, target) + nn.BCEWithLogitsLoss()(pred, target)
    if loss_type == 'dice_bce_mc':
        loss = DiceLoss(multiclass=True)(pred, target) + \
            nn.CrossEntropyLoss()(pred, target.long())
    if loss_type == 'dice_score':
        loss = DiceLoss().dice_score(pred, target)
    if loss_type == 'log_cosh_dice_loss':
        x = DiceLoss()(pred, target)
        loss = torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)
    if loss_type == 'dice_score_mc':
        loss = DiceLoss(multiclass=True).dice_score_mc(pred, target)
    return loss
