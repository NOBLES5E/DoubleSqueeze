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
from compression import *

@ray.remote
class GradientParameterServer(object):
    def __init__(self):
        self.compensation_buffer = None
        pass

    def get_ps_gradient(self, *gradients):
        # print('server', gradients)
        return torch.sum(torch.stack(gradients), dim=0)

    def get_quantized_gradient(self, *gradients_inputs):
        gradients = list(map(lambda x: decompress_gradient(*x), gradients_inputs))
        return tenary_compress_gradient(self.get_ps_gradient(*gradients))[:4]

    def get_quantized_gradient_no_requant(self, *gradients_inputs):
        gradients = list(map(lambda x: decompress_gradient(*x), gradients_inputs))
        return self.get_ps_gradient(*gradients)

    def get_random_quantized_gradient(self, *gradients_inputs):
        gradients = list(map(lambda x: decompress_gradient(*x), gradients_inputs))
        return randomized_tenary_compress_gradient(self.get_ps_gradient(*gradients))[:4]

    def get_all_random_quantized_gradient(self, *gradients_inputs):
        return gradients_inputs

    def get_compensated_quantized_gradient(self, *gradients_inputs):
        gradients = list(map(lambda x: decompress_gradient(*x), gradients_inputs))
        gradient = self.get_ps_gradient(*gradients)
        if self.compensation_buffer is None:
            self.compensation_buffer = torch.zeros_like(gradient)
        compensated_gradient = gradient + self.compensation_buffer
        result = tenary_compress_gradient(compensated_gradient)
        self.compensation_buffer = result[-1]
        return result[:-1]

    def get_compensated_top_k_gradient(self, *gradients_inputs):
        gradients = list(map(lambda x: decompress_top_k_gradient(*x), gradients_inputs))
        gradient = self.get_ps_gradient(*gradients)
        if self.compensation_buffer is None:
            self.compensation_buffer = torch.zeros_like(gradient)
        compensated_gradient = gradient + self.compensation_buffer
        result = top_k_compress_gradient(compensated_gradient)
        self.compensation_buffer = result[-1]
        return result[:-1]

    def get_topk_gradient(self, *gradients_inputs):
        gradients = list(map(lambda x: decompress_top_k_gradient(*x), gradients_inputs))
        gradient = self.get_ps_gradient(*gradients)
        return gradient
