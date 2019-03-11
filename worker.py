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

@ray.remote(num_gpus=1)
class Worker(object):
    def __init__(self, worker_index, state_dict, lr):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, ray.get_gpu_ids()))
        torch.manual_seed(worker_index)
        print('Worker', worker_index, 'using gpu', ray.get_gpu_ids())
        self.worker_index = worker_index
        self.trainloader, _ = get_dataloaders()
        self.net = model.PreActResNet18()
        self.net.load_state_dict(state_dict)
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0)
        self.criterion = nn.CrossEntropyLoss()
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, 20, gamma=0.1)

        self.train_loss = 0
        self.correct = 0
        self.total = 0
        self.epoch = 0

        self.start_time = time.time()

        self.data_iter = enumerate(self.trainloader)

        self.compensation_buffer = torch.zeros_like(model_utils.model_to_flatten_parameters(self.net.to('cpu')))

    def get_state_dict(self):
        return self.net.state_dict()


    def compute_local_gradients(self):
        # print('worker', list(self.net.parameters())[5])
        self.net.to('cuda')
        self.net.train()
        try:
            self.batch_idx, (inputs, targets) = next(self.data_iter)
        except StopIteration:
            self.data_iter = enumerate(self.trainloader)
            self.batch_idx, (inputs, targets) = next(self.data_iter)

            self.train_loss = 0
            self.correct = 0
            self.total = 0
            self.epoch += 1

            self.lr_scheduler.step()

            model_utils.checkpoint(self.net, "./results", "epoch-" + str(self.epoch))

        self.inputs, self.targets = inputs.to('cuda'), targets.to('cuda')
        self.optimizer.zero_grad()
        self.outputs = self.net(self.inputs)
        self.loss = self.criterion(self.outputs, self.targets)
        self.loss.backward()
        return model_utils.model_to_flatten_gradients(self.net.to('cpu'))

    def compute_quantized_gradients(self):
        gradient = self.compute_local_gradients()
        return tenary_compress_gradient(gradient)[:4]

    def compute_top_k_gradients(self):
        gradient = self.compute_local_gradients()
        return top_k_compress_gradient(gradient)[:-1]

    def compute_top_k_gradients_compensated(self):
        gradient = self.compute_local_gradients()
        compensated_gradient = gradient + self.compensation_buffer
        result = top_k_compress_gradient(compensated_gradient)
        self.compensation_buffer = result[-1]
        return result[:-1]

    def apply_top_k_gradient(self, inputs):
        self.apply_ps_gradient(decompress_top_k_gradient(*inputs))

    def compute_random_quantized_gradients(self):
        gradient = self.compute_local_gradients()
        return randomized_tenary_compress_gradient(gradient)[:4]

    def compute_quantized_gradients_compensated(self):
        gradient = self.compute_local_gradients()
        compensated_gradient = gradient + self.compensation_buffer
        result = tenary_compress_gradient(compensated_gradient)
        self.compensation_buffer = result[-1]
        return result[:-1]

    def apply_ps_gradient(self, gradient):
        self.net.to('cuda')
        gradient_parameters = model_utils.flatten_parameters_to_model(gradient.to('cuda'), self.net)
        for param, grad in zip(self.net.parameters(), gradient_parameters):
            param.grad.data.set_(grad)
        self.optimizer.step()
        self.train_loss += self.loss.item()
        _, predicted = self.outputs.max(1)
        self.total += self.targets.size(0)
        self.correct += predicted.eq(self.targets).sum().item()
        print('Time %d | Epoch %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (time.time() - self.start_time, self.epoch, self.train_loss/(self.batch_idx+1), 100.*self.correct/self.total, self.correct, self.total))


    def apply_quantized_gradient(self, inputs):
        gradient = decompress_gradient(*inputs)
        self.apply_ps_gradient(gradient)
