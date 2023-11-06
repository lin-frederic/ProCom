import torch
from typing import Tuple
import numpy as np

def unravel_index(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).
    """

    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    return coord.flip(-1)


def iou(maskA, maskB):
    # maskA and maskB are (H,W) numpy arrays
    assert maskA.shape == maskB.shape
    # convert to boolean
    maskA = maskA>0
    maskB = maskB>0
    intersection = np.logical_and(maskA, maskB)
    union = np.logical_or(maskA, maskB)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# 2 segmentation losses that could be used as a replacement for IoU

def focal_loss(inputs : np.ndarray,targets : np.ndarray, alpha : float = 1., gamma : float = 2.):
    # inputs and targets are numpy arrays of the same shape
    
    inputs = inputs.flatten()
    targets = targets.flatten()
    
    inputs = np.clip(inputs,1e-6,1-1e-6)
    targets = np.clip(targets,1e-6,1-1e-6)
    
    BCE = - (targets * np.log(inputs) + (1-targets) * np.log(1-inputs))
    
    BCE_EXP = np.exp(BCE)
    
    loss = np.mean(alpha * (1-BCE_EXP)**gamma * BCE)
    
    return loss

def dice_loss(inputs : np.ndarray,targets : np.ndarray, smooth : float = 1.):
    # inputs and targets are numpy arrays of the same shape
    
    inputs = inputs.flatten()
    targets = targets.flatten()

    
    intersection = np.sum(inputs * targets)
    union = np.sum(inputs) + np.sum(targets)
    
    loss = (2. * intersection + smooth) / (union + smooth)
    
    return 1 - loss
    
    