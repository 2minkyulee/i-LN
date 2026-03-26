import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MaskedL1Loss(nn.Module):
    
    def __init__(self, loss_weight=1.0, reduction='mean', topk_percentage=50):
        super(MaskedL1Loss, self).__init__()
        if reduction not in ['mean']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean')
        
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.topk_percentage = topk_percentage/100
    
    def forward(self, pred, target, weight=None, **kwargs):
        """
        Use only n% of the pixels with the highest loss values.
        Averaging is done over all pixels.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        
        
        # Compute the element-wise L1 loss
        loss_map = torch.abs(pred - target)
        
        
        # get mask
        with torch.no_grad():
            if self.topk_percentage > 0:
                # Compute the threshold value for the top-k percentage without reshaping
                threshold = torch.quantile(loss_map, 1 - self.topk_percentage)
                mask = loss_map >= threshold
            else:
                # # If topk_percentage is zero, create an all-zero mask
                # mask = torch.zeros_like(loss_map, dtype=torch.bool)
                raise ValueError('Top-k percentage must be greater than zero')
        
        
        # Apply the mask to the loss map
        masked_loss_map = loss_map * mask.float()
        
        # Apply the optional weight if provided
        if weight is not None:
            # masked_loss_map = masked_loss_map * weight
            raise ValueError('Weighted Masked L1 Loss is not supported')
        
        # Compute the final loss based on the reduction method
        loss = masked_loss_map.sum() / loss_map.numel()
        
        # Apply the loss weight
        return loss * self.loss_weight

