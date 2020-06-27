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

from utils import clamp, attack_fgsm, attack_pgd

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--data-dir', default='../mnist-data', type=str)
    parser.add_argument('--fname', type=str)
    parser.add_argument('--ename', default='output', type=str, help='experiment name for logging')
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'none'])
    parser.add_argument('--epsilon', default=0.3, type=float)
    parser.add_argument('--attack-iters', default=50, type=int)
    parser.add_argument('--alpha', default=1e-2, type=float)
    parser.add_argument('--restarts', default=10, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--norm', default='linf', type=str, choices=['linf', 'l1', 'l2', 'l2-scaled'])
    parser.add_argument('--init', default='random', type=str, choices=['random', 'zero'])
    return parser.parse_args()


def main():
    args = get_args()

    logging.basicConfig(
        format='[%(asctime)s %(filename)s %(name)s %(levelname)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        filename=f'{args.ename}.log',
        level=logging.DEBUG)

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

    deltas = None
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
                delta = attack_pgd(model, X, y, args.epsilon, args.alpha, args.attack_iters, args.restarts, args.norm, args.init)
                # store deltas for viz
                if deltas != None:
                    deltas = torch.cat((deltas, delta))
                else:
                    deltas = delta
            elif args.attack == 'fgsm':
                delta = attack_fgsm(model, X, y, args.epsilon, args.alpha, args.norm, args.init)
            
            with torch.no_grad():   
                #plot(X, delta, y)
                output = model(X + delta)
                loss = F.cross_entropy(output, y)
                total_loss += loss.item() * y.size(0)
                total_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)

    logger.info('Test Loss: %.4f, Acc: %.4f', total_loss/n, total_acc/n)
    if deltas != None:
        torch.save(deltas.cpu(), args.fname+'_eval_deltas')


def plot(data, delta, label):
        # Reshape the array into 28 x 28 array (2-dimensional array)
        #data = data.cpu().detach()[0].reshape((28, 28))
        delta = delta.cpu().detach().numpy().reshape((delta.shape[0],-1))
        normal = np.ones(delta.shape[1])
        normal[0:2] = 0

        projected = delta - delta * normal * normal.transpose()
        # Plot
        #plt.title('Label is {label}'.format(label=label[0]))
        #plt.imshow(data+delta, cmap='gray', vmin=0., vmax=1.0)
        plt.scatter(projected[:,0], projected[:,1])
        plt.show()

if __name__ == "__main__":
    main()
