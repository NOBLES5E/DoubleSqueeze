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
from ps import *
from worker import *

def get_dataloaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
       ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
       ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader



def test(net, testloader):
    net.eval()
    net.to('cuda')
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('TEST | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the synchronous parameter " "server example.")
    parser.add_argument("--num-workers", default=8, type=int, help="The number of workers to use.")
    parser.add_argument("--redis-address", default=None, type=str, help="The Redis address of the cluster.")
    parser.add_argument("--mode", default=None, type=str, help="original, quantized, quantized_compensate")
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')

    args = parser.parse_args()

    # download dataset
    _, testloader = get_dataloaders()

    ray.init(redis_address=args.redis_address, num_gpus=8)

    # Create a parameter server.
    net = model.PreActResNet18()
    ps = GradientParameterServer.remote()

    # Create workers.
    workers = [Worker.remote(worker_index, net.state_dict(), args.lr) for worker_index in range(args.num_workers)]

    i = 0
    while True:
        # Compute and apply gradients.
        if args.mode == "original":
            if i == 0:
                print("USING ORIGINAL PS ALGORITHM")
            gradients = [worker.compute_local_gradients.remote() for worker in workers]
            ps_gradient = ps.get_ps_gradient.remote(*gradients)
            worker_result_ids = [worker.apply_ps_gradient.remote(ps_gradient) for worker in workers]
        elif args.mode == "quantized":
            if i == 0:
                print("USING QUANTIZED ALGORITHM")
            gradients = [worker.compute_quantized_gradients.remote() for worker in workers]
            ps_gradient = ps.get_quantized_gradient.remote(*gradients)
            worker_result_ids = [worker.apply_quantized_gradient.remote(ps_gradient) for worker in workers]
        elif args.mode == "compensated":
            if i == 0:
                print("USING COMPENSATED QUANTIZED ALGORITHM")
            gradients = [worker.compute_quantized_gradients_compensated.remote() for worker in workers]
            ps_gradient = ps.get_compensated_quantized_gradient.remote(*gradients)
            worker_result_ids = [worker.apply_quantized_gradient.remote(ps_gradient) for worker in workers]
        elif args.mode == "compensated_worker":
            if i == 0:
                print("USING COMPENSATED QUANTIZED ALGORITHM ONLY ON WORKERS")
            gradients = [worker.compute_quantized_gradients_compensated.remote() for worker in workers]
            ps_gradient = ps.get_quantized_gradient.remote(*gradients)
            worker_result_ids = [worker.apply_quantized_gradient.remote(ps_gradient) for worker in workers]
        elif args.mode == "compensated_worker_norequant":
            if i == 0:
                print("USING COMPENSATED QUANTIZED ALGORITHM ONLY ON WORKERS WITHOUT RE-QUANTIZATION")
            gradients = [worker.compute_quantized_gradients_compensated.remote() for worker in workers]
            ps_gradient = ps.get_quantized_gradient_no_requant.remote(*gradients)
            worker_result_ids = [worker.apply_ps_gradient.remote(ps_gradient) for worker in workers]
        elif args.mode == "randomized_quantization":
            if i == 0:
                print("USING RANDOMIZED QUANTIZATION ALGORITHM")
            gradients = [worker.compute_random_quantized_gradients.remote() for worker in workers]
            ps_gradient = ps.get_random_quantized_gradient.remote(*gradients)
            worker_result_ids = [worker.apply_quantized_gradient.remote(ps_gradient) for worker in workers]
        elif args.mode == "randomized_quantization_no_requant":
            if i == 0:
                print("USING RANDOMIZED QUANTIZATION ALGORITHM WITHOUT RE-QUANTIZATION")
            gradients = [worker.compute_random_quantized_gradients.remote() for worker in workers]
            ps_gradient = ps.get_quantized_gradient_no_requant.remote(*gradients)
            worker_result_ids = [worker.apply_ps_gradient.remote(ps_gradient) for worker in workers]
        elif args.mode == "topk":
            if i == 0:
                print("USING PLAIN TOPK ALGORITHM")
            gradients = [worker.compute_top_k_gradients.remote() for worker in workers]
            ps_gradient = ps.get_topk_gradient.remote(*gradients)
            worker_result_ids = [worker.apply_ps_gradient.remote(ps_gradient) for worker in workers]
        elif args.mode == "topk_compensated":
            if i == 0:
                print("USING COMPENSATED TOPK ALGORITHM")
            gradients = [worker.compute_top_k_gradients_compensated.remote() for worker in workers]
            ps_gradient = ps.get_compensated_top_k_gradient.remote(*gradients)
            worker_result_ids = [worker.apply_top_k_gradient.remote(ps_gradient) for worker in workers]
        elif args.mode == "topk_worker_compensated":
            if i == 0:
                print("USING COMPENSATED TOPK ALGORITHM ONLY ON WORKER")
            gradients = [worker.compute_top_k_gradients_compensated.remote() for worker in workers]
            ps_gradient = ps.get_topk_gradient.remote(*gradients)
            worker_result_ids = [worker.apply_ps_gradient.remote(ps_gradient) for worker in workers]


        if i % 100 == 0:
            # Evaluate the current models.
            net.load_state_dict(ray.get(workers[0].get_state_dict.remote()))
            test(net, testloader)

        i += 1
