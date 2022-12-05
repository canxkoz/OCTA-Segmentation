import torch
import torch.nn as nn
import torch.nn.functional as F

# class DiceBCELoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceBCELoss, self).__init__()

#     def forward(self, inputs, targets, smooth=1):
        
#         #comment out if your model contains a sigmoid or equivalent activation layer
#         # inputs = F.sigmoid(inputs)       
        
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         intersection = (inputs * targets).sum()                            
#         dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
#         BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
#         Dice_BCE = BCE + dice_loss
        
#         return Dice_BCE


def calc_loss(pred, target, metrics, bce_weight=0.5, loss_type='mse'):
    if loss_type == 'bce':
        #bce_logits_loss = nn.BCEWithLogitsLoss()
        loss =  nn.BCEWithLogitsLoss()(pred, target)
    elif loss_type == 'mse':
        loss = nn.MSELoss()(pred, target)
    elif loss_type == 'rmse':
        mse = nn.MSELoss()(pred, target)
        loss = torch.sqrt(mse)
    elif loss_type == 'l1loss':
        loss = nn.L1Loss()(pred, target)

    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss