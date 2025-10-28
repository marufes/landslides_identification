########################################################################################################################
# 训练验证过程相关API
########################################################################################################################

import torch
from torch import nn
import torch.nn.functional
import distributed_utils as utils
# With this line:
from loss_function import build_target, Focal_Loss, Dice_loss, Weighted_CE_Loss, CE_Loss
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

# def find_boundary(target: torch.Tensor):
#     """
#     Finds boundary pixels (1s with at least one 0 neighbor) in a binary mask.
#     Also prints total number of 1s and boundary pixels.

#     Args:
#         target (torch.Tensor): Tensor of shape (H, W) or (B, H, W) with values {0,1}.
#     Returns:
#         torch.Tensor: Boundary map (same shape as target) with 1 at boundary pixels, else 0.
#     """

#     # Ensure float tensor and 4D shape (B, 1, H, W)
#     if target.dim() == 2:
#         target = target.unsqueeze(0).unsqueeze(0)
#     elif target.dim() == 3:
#         target = target.unsqueeze(1)
#     target = target.float()

#     # Define 3×3 kernel to check all 8 neighbors
#     kernel = torch.ones((1, 1, 3, 3), device=target.device)
#     kernel[:, :, 1, 1] = 0  # exclude center pixel

#     # Count number of 1 neighbors per pixel
#     neighbor_count = F.conv2d(target, kernel, padding=1)

#     # Boundary condition: pixel=1 and not all neighbors are 1
#     boundary = (target == 1) & (neighbor_count < 8)

#     # Flatten to original shape
#     boundary = boundary.squeeze(1).long()

#     # Compute and print stats
#     total_ones = int(target.sum().item())
#     boundary_count = int(boundary.sum().item())

#     print(f"Total 1s in target: {total_ones}")
#     print(f"Boundary pixels: {boundary_count}")

#     return boundary


import torch
import torch.nn.functional as F

def find_boundary(target: torch.Tensor):
    """
    Finds bidirectional boundary pixels between 0 and 1 regions.
    A pixel is boundary if:
      - It is 1 and has at least one 0 neighbor, OR
      - It is 0 and has at least one 1 neighbor.

    Args:
        target (torch.Tensor): Binary mask of shape (H, W) or (B, H, W).
    Returns:
        torch.Tensor: Boundary map (same shape as target) with 1 for boundary pixels, else 0.
    """

    # Ensure tensor is float and has shape (B,1,H,W)
    if target.dim() == 2:
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.dim() == 3:
        target = target.unsqueeze(1)
    target = target.float()

    # 3x3 kernel for 8-connected neighborhood
    kernel = torch.ones((1, 1, 3, 3), device=target.device)
    kernel[:, :, 1, 1] = 0  # exclude center

    # Count neighboring 1s for each pixel
    neighbor_count = F.conv2d(target, kernel, padding=1)

    # --- Boundary condition ---
    # Case 1: pixel = 1 and has at least one 0 neighbor
    boundary_1 = (target == 1) & (neighbor_count < 8)
    # Case 2: pixel = 0 and has at least one 1 neighbor
    boundary_0 = (target == 0) & (neighbor_count > 0)

    # Combine both boundaries
    boundary = (boundary_1 | boundary_0).squeeze(1).long()

    # --- Statistics ---
    total_ones = int(target.sum().item())
    boundary_count = int(boundary.sum().item())

    print(f"Total 1s in target: {total_ones}")
    print(f"Boundary pixels (both 0↔1 sides): {boundary_count}")

    return boundary


def criterion(inputs, target, num_classes: int = 2, focal_loss: bool = False, dice_loss: bool = False):
    losses = {}
    
    # img = target[0].detach().cpu().numpy()
    # plt.figure(figsize=(4, 4))
    # print(img)
    # plt.title("Ground Truth Mask")
    # plt.imshow(img, cmap='gray')
    # plt.axis('off')
    # plt.show()

    print("Finding boundary pixels for ground truth")
    boundary_ref = find_boundary(target)
    weight_map = torch.ones_like(target).float()
    weight_map[boundary_ref == 1] = 1.2 # Example: double weight on boundary pixels
    weight_map = weight_map / weight_map.mean()
    
    # print("Finding boundary pixels for predictions")
    # boundary_pred = find_boundary(inputs)

    for name, x in inputs.items():
            if focal_loss:
                # loss = Focal_Loss(x, target, ignore_index=255)
                loss = Focal_Loss(x, target, weight_map=weight_map, ignore_index=255)
            else:
                #loss = CE_Loss(x, target, ignore_index=255)
                loss = Weighted_CE_Loss(x, target, weight_map=weight_map, ignore_index=255)
    
            if dice_loss:
                dice_target = build_target(target, num_classes, ignore_index=255)
                dice_loss = Dice_loss(x, dice_target)
                loss = loss + dice_loss
    
            losses[name] = loss
    return losses['out']


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Eval:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target, num_classes=2, focal_loss=False, dice_loss=False)

            output1 = output['out']

            confmat.update(target.flatten(), output1.argmax(1).flatten())

            metric_logger.update(loss=loss.item())
    return metric_logger.meters["loss"].global_avg, confmat

def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=100, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target, num_classes=2, focal_loss=True, dice_loss=True)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):

        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)

            return warmup_factor * (1 - alpha) + alpha
        else:

            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
