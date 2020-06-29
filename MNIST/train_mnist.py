import argparse
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from mnist_net import mnist_net

from utils import clamp, attack_fgsm, attack_pgd

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--data-dir', default='../mnist-data', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--attack', default='fgsm', type=str, choices=['none', 'pgd', 'fgsm'])
    parser.add_argument('--attack-type', default='single', choices=['single', 'max', 'avg', 'random', 'avg_loss'])
    parser.add_argument('--epsilon', default=0.3, type=float)
    parser.add_argument('--alpha', default=0.375, type=float)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--attack-iters', default=40, type=int)
    parser.add_argument('--lr-max', default=5e-3, type=float)
    parser.add_argument('--lr-type', default='cyclic')
    parser.add_argument('--fname', default='mnist_model', type=str)
    parser.add_argument('--ename', default='output', type=str, help='experiment name for logging')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--norm', default='linf', type=str, choices=['linf', 'l1', 'l2', 'l2-scaled'])
    parser.add_argument('--init', default='random', type=str, choices=['random', 'zero'])
    return parser.parse_args()


def main():
    args = get_args()

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        filename=f'{args.ename}.log',
        level=logging.DEBUG)

    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    mnist_train = datasets.MNIST("../mnist-data", train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True)

    model = mnist_net().cuda()
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr_max)
    if args.lr_type == 'cyclic': 
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2//5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_type == 'flat': 
        lr_schedule = lambda t: args.lr_max
    else:
        raise ValueError('Unknown lr_type')

    criterion = nn.CrossEntropyLoss()

    logger.info('Epoch \t Time \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        deltas = None
        gradients = None
        selected_attack = []
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            lr = lr_schedule(epoch + (i+1)/len(train_loader))
            opt.param_groups[0].update(lr=lr)

            if args.attack == 'fgsm':
                if args.attack_type == 'random':
                    norms_list = ['linf', 'l2-scaled', 'l1']
                    epsilon_list = [0.3, 6.5, 180.0]
                    alpha_list = [0.375, 2.5, 225.0]
                    curr_norm =  np.random.randint(len(norms_list))
                    selected_attack.append(norms_list[curr_norm])
                    delta = attack_fgsm(model, X, y, epsilon_list[curr_norm], alpha_list[curr_norm], norms_list[curr_norm], args.init)
                elif args.attack_type == 'max' or args.attack_type == "avg" or args.attack_type == "avg_loss":
                    norms_list = ['linf', 'l2-scaled', 'l1']
                    epsilon_list = [0.3, 6.5, 180.0]
                    alpha_list = [0.375, 2.5, 225.0]
                    delta_linf = attack_fgsm(model, X, y, epsilon_list[0], alpha_list[0], norms_list[0], args.init)
                    delta_l1 = attack_fgsm(model, X, y, epsilon_list[1], alpha_list[1], norms_list[1], args.init)
                    delta_l2 = attack_fgsm(model, X, y, epsilon_list[2], alpha_list[2], norms_list[2], args.init)
                else:
                    delta = attack_fgsm(model, X, y, args.epsilon, args.alpha, args.norm, args.init)
                #if deltas != None:
                #    deltas = torch.cat((deltas, delta), dim=0)
                #else:
                #    deltas = delta
            elif args.attack == 'none':
                delta = torch.zeros_like(X)
            elif args.attack == 'pgd':
                if args.attack_type == 'random':
                    norms_list = ['linf', 'l2-scaled', 'l1']
                    epsilon_list = [0.3, 6.5, 180.0]
                    alpha_list = [0.01, 0.03, 6.0]
                    curr_norm =  np.random.randint(len(norms_list))
                    selected_attack.append(norms_list[curr_norm])
                    delta = attack_pgd(model, X, y, epsilon_list[curr_norm], alpha_list[curr_norm], args.attack_iters, args.restarts, norms_list[curr_norm], args.init)
                elif args.attack_type == 'max' or args.attack_type == "avg" or args.attack_type == "avg_loss":
                    norms_list = ['linf', 'l2-scaled', 'l1']
                    epsilon_list = [0.3, 6.5, 180.0]
                    alpha_list = [0.01, 0.03, 6.0]
                    delta_linf = attack_pgd(model, X, y, epsilon_list[0], alpha_list[0], args.attack_iters, args.restarts, norms_list[0], args.init)
                    delta_l1 = attack_pgd(model, X, y, epsilon_list[1], alpha_list[1], args.attack_iters, args.restarts, norms_list[1], args.init)
                    delta_l2 = attack_pgd(model, X, y, epsilon_list[2], alpha_list[2], args.attack_iters, args.restarts, norms_list[2], args.init)
                else:
                    delta = attack_pgd(model, X, y, args.epsilon, args.alpha, args.attack_iters, args.restarts, args.norm, args.init)
            
            if args.attack_type == "single" or args.attack_type == "random":
                output = model(torch.clamp(X + delta, 0, 1))
                loss = criterion(output, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            elif args.attack_type == 'max':
                output_linf = model(torch.clamp(X + delta_linf[:X.size(0)], 0, 1))
                output_l1 = model(torch.clamp(X + delta_l1[:X.size(0)], 0, 1))
                output_l2 = model(torch.clamp(X + delta_l2[:X.size(0)], 0, 1))
                loss_linf = criterion(output_linf, y)
                loss_l1 = criterion(output_l1, y)
                loss_l2 = criterion(output_l2, y)

                acc_linf = (output_linf.max(1)[1] == y).sum().item()
                acc_l1 = (output_l1.max(1)[1] == y).sum().item()
                acc_l2 = (output_l2.max(1)[1] == y).sum().item()

                loss_list = [loss_l1, loss_l2, loss_linf]
                acc_list = [acc_l1, acc_l2, acc_linf]

                delta_list = [delta_l1, delta_l2, delta_linf]
                max_loss =  max(loss_list)
                max_loss_index = loss_list.index(max_loss)
                max_delta = delta_list[max_loss_index]

                output = model(torch.clamp(X + max_delta[:X.size(0)], 0, 1))
                loss = criterion(output, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
            elif args.attack_type == "avg_loss":
                output = model(torch.clamp(X + delta_linf[:X.size(0)], 0, 1))
                loss_linf = criterion(output, y)
                output_l1 = model(torch.clamp(X + delta_l1[:X.size(0)], 0, 1))
                loss_l1 = criterion(output_l1, y)
                output_l2 = model(torch.clamp(X + delta_l2[:X.size(0)], 0, 1))
                loss_l2 = criterion(output_l2, y)

                loss = (loss_linf + loss_l1 + loss_l2)/3

                opt.zero_grad()
                # with amp.scale_loss(loss, opt) as scaled_loss:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()

            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        train_time = time.time()
        logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f',
            epoch, train_time - start_time, lr, train_loss/train_n, train_acc/train_n)
        torch.save(model.state_dict(), args.fname)
        if deltas != None:
            torch.save(deltas.cpu(), args.fname+'_deltas')
        if gradients != None:
            torch.save(gradients.cpu(), args.fname+'_gradients')
        #np.array(selected_attack)


if __name__ == "__main__":
    main()
