import torch
import torch.nn as nn
import torch.nn.functional as F


import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import ray

from persia_pytorch_toolkit import model_utils
from importlib import reload
import model

def tenary_compress_gradient(gradient: torch.Tensor):
    l2 = gradient.norm()
    sign_tensor = gradient.sign()
    sign_tensor_l2 = sign_tensor.norm()
    scale = l2 / sign_tensor_l2
    difference = gradient - sign_tensor * scale
    sign_numpy = sign_tensor.numpy().astype(int)
    greater = np.packbits(sign_numpy > 0)
    lower = np.packbits(sign_numpy < 0)
    length = gradient.nelement()
    return greater, lower, scale, length, difference


def top_k_compress_gradient(gradient: torch.Tensor):
    l2 = gradient.norm()

    _, topk_tensor_indices = gradient.abs().topk(k=300000) # 11171146
    topk_tensor = torch.index_select(gradient, 0, topk_tensor_indices)

    # scale = l2 / topk_tensor.norm()
    # topk_tensor *= scale

    sparse_topk_tensor = torch.sparse_coo_tensor(topk_tensor_indices.unsqueeze(0),
                                                 topk_tensor,
                                                 size=gradient.size())

    difference = gradient - sparse_topk_tensor
    return topk_tensor_indices.unsqueeze(0), topk_tensor, gradient.size(), difference

def decompress_top_k_gradient(indices, values, size):
    return torch.sparse_coo_tensor(indices, values, size).to_dense()

def randomized_tenary_compress_gradient(gradient: torch.Tensor):
    l2 = gradient.norm()
    sign_tensor = gradient.sign()

    scale_tensor = gradient.abs()
    scale_tensor /= scale_tensor.max()
    random_tensor = torch.rand_like(scale_tensor)
    mask_tensor = scale_tensor >= random_tensor

    sign_tensor *= mask_tensor.float()

    sign_tensor_l2 = sign_tensor.norm()
    scale = l2 / sign_tensor_l2
    difference = gradient - sign_tensor * scale
    sign_numpy = sign_tensor.numpy().astype(int)
    greater = np.packbits(sign_numpy > 0)
    lower = np.packbits(sign_numpy < 0)
    length = gradient.nelement()
    return greater, lower, scale, length, difference


def decompress_gradient(greater, lower, scale, length):
    return scale * ( torch.from_numpy(np.unpackbits(greater).astype(np.float32)[:length]) - torch.from_numpy(np.unpackbits(lower).astype(np.float32)[:length]) )
