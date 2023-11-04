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
