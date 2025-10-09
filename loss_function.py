########################################################################################################################
# loss函数
########################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn

def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """build target for dice coefficient"""
    dice_target = target.clone()
    # if ignore_index >= 0:
    ignore_mask = torch.eq(target, ignore_index)
    dice_target[ignore_mask] = 2
    # [N, H, W] -> [N, H, W, C]
    dice_target = nn.functional.one_hot(dice_target, num_classes+1).float()
    # else:
    #     dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target

# def CE_Loss(inputs, target, ignore_index: int = -100):
#     n, c, h, w = inputs.size()
#     nt, ht, wt = target.size()

#     temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
#     temp_target = target.view(-1)

#     CE_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)(temp_inputs, temp_target)
#     return CE_loss

def Weighted_CE_Loss(inputs, target, weight_map=None, ignore_index: int = -100):
    n, c, h, w = inputs.size()

    # Flatten predictions and targets
    temp_inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, c)  # (N*H*W, C)
    temp_target = target.view(-1)                                      # (N*H*W)

    # Compute per-pixel loss without reduction
    ce = F.cross_entropy(temp_inputs, temp_target, ignore_index=ignore_index, reduction='none')

    if weight_map is not None:
        weight_map_flat = weight_map.view(-1)
        ce = ce * weight_map_flat

    # Mask out ignored pixels
    valid_mask = (temp_target != ignore_index)
    ce = ce[valid_mask]

    return ce.mean()


# def Focal_Loss(inputs, target, ignore_index: int = -100, alpha=0.5, gamma=2):
#     n, c, h, w = inputs.size()
#     nt, ht, wt = target.size()

#     temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
#     temp_target = target.view(-1)

#     logpt = -nn.CrossEntropyLoss(ignore_index=ignore_index)(temp_inputs, temp_target)

#     pt = torch.exp(logpt)
#     if alpha is not None:
#         logpt *= alpha
#     loss = -((1 - pt) ** gamma) * logpt
#     focal_loss = loss.mean()
#     return focal_loss

def Focal_Loss(inputs, target, weight_map=None, ignore_index: int = -100, alpha=0.5, gamma=2.0):
    """
    Boundary-aware focal loss for segmentation.
    
    Args:
        inputs (torch.Tensor): [N, C, H, W] logits (unnormalized).
        target (torch.Tensor): [N, H, W] ground truth labels.
        weight_map (torch.Tensor, optional): [N, H, W], pixel-wise weights (e.g., boundary emphasis).
        ignore_index (int): label to ignore in loss computation.
        alpha (float): class balancing factor.
        gamma (float): focusing parameter.
    """
    n, c, h, w = inputs.size()

    # Flatten inputs and targets
    inputs_flat = inputs.permute(0, 2, 3, 1).contiguous().view(-1, c)  # (N*H*W, C)
    targets_flat = target.view(-1)                                    # (N*H*W)

    # Compute per-pixel CE loss (no reduction!)
    ce_loss = F.cross_entropy(inputs_flat, targets_flat, ignore_index=ignore_index, reduction='none')

    # Get probabilities for the true class
    pt = torch.exp(-ce_loss)

    # Apply alpha if specified
    if alpha is not None:
        ce_loss = alpha * ce_loss

    # Focal loss term
    focal_loss = ((1 - pt) ** gamma) * ce_loss

    # If boundary weights are provided, apply them
    if weight_map is not None:
        weight_map_flat = weight_map.view(-1)
        focal_loss = focal_loss * weight_map_flat

    # Mask out ignored pixels explicitly
    valid_mask = (targets_flat != ignore_index)
    focal_loss = focal_loss[valid_mask]

    return focal_loss.mean()

def Dice_loss(inputs, dice_target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = dice_target.size()

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = dice_target.view(n, -1, ct)

    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[...,:-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss


