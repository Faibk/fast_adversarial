import argparse
import logging
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

from mnist_net import mnist_net

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s %(filename)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    filename='output.log',
    level=logging.DEBUG)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_fgsm(model, X, y, epsilon, norm):
    delta = torch.zeros_like(X, requires_grad=True)
    output = model(X + delta)
    loss = F.cross_entropy(output, y)
    loss.backward()
    grad = delta.grad.detach()  
    delta.data = epsilon * torch.sign(grad)
    return delta.detach()


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, norm):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).cuda()
        delta.data = clamp(delta, 0-X, 1-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)[0]
            if len(index) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            if norm == 'linf':
                d = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            elif norm == 'l2': #using sign of gradient --> limits directions of gradient
                d = delta + alpha * torch.sign(grad)
                d_flat = d.view(d.size(0),-1)
                norm = d_flat.norm(p=2,dim=1).clamp(min=epsilon).view(d.size(0),1,1,1)
                d *=  epsilon / norm
            elif norm == 'l2-scaled':
                d = delta + alpha * grad / grad.view(grad.shape[0], -1).norm(dim=1)[:,None,None,None]
                d_flat = d.view(d.size(0),-1)
                norm = d_flat.norm(p=2,dim=1).clamp(min=epsilon).view(d.size(0),1,1,1)
                d *=  epsilon / norm
            d = clamp(d, 0-X, 1-X)
            delta.data[index] = d[index]
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--data-dir', default='../mnist-data', type=str)
    parser.add_argument('--fname', type=str)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'none'])
    parser.add_argument('--epsilon', default=0.3, type=float)
    parser.add_argument('--attack-iters', default=50, type=int)
    parser.add_argument('--alpha', default=1e-2, type=float)
    parser.add_argument('--restarts', default=10, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--norm', default='linf', type=str, choices=['linf', 'l1', 'l2', 'l2-scaled'])
    return parser.parse_args()


def main():
    args = get_args()
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    mnist_test = datasets.MNIST("../mnist-data", train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=args.batch_size, shuffle=False)

    model = mnist_net().cuda()
    checkpoint = torch.load(args.fname)
    model.load_state_dict(checkpoint)
    model.eval()

    total_loss = 0
    total_acc = 0
    n = 0

    if args.attack == 'none':
        with torch.no_grad():
            for i, (X, y) in enumerate(test_loader):
                X, y = X.cuda(), y.cuda()
                output = model(X)
                loss = F.cross_entropy(output, y)
                total_loss += loss.item() * y.size(0)
                total_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)
    else:
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            if args.attack == 'pgd':
                delta = attack_pgd(model, X, y, args.epsilon, args.alpha, args.attack_iters, args.restarts, args.norm)
            elif args.attack == 'fgsm':
                delta = attack_fgsm(model, X, y, args.epsilon, args.norm)
            with torch.no_grad():   
                #plot(X, delta, y)
                output = model(X + delta)
                loss = F.cross_entropy(output, y)
                total_loss += loss.item() * y.size(0)
                total_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)

    logger.info('Test Loss: %.4f, Acc: %.4f', total_loss/n, total_acc/n)


def plot(data, delta, label):
        # Reshape the array into 28 x 28 array (2-dimensional array)
        #data = data.cpu().detach()[0].reshape((28, 28))
        delta = delta.cpu().detach().numpy().reshape((delta.shape[0],-1))
        normal = np.ones(delta.shape[1])
        normal[0:2] = 0

        projected = delta - delta * normal * normal.transpose()
        print(projected[1])
        # Plot
        #plt.title('Label is {label}'.format(label=label[0]))
        #plt.imshow(data+delta, cmap='gray', vmin=0., vmax=1.0)
        plt.scatter(projected[:,0], projected[:,1])
        plt.show()

if __name__ == "__main__":
    main()
