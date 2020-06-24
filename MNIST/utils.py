import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_fgsm(model, X, y, epsilon, alpha, norm, init):
    return attack_pgd(model, X, y, epsilon, alpha, 1, 1, norm, init)

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, norm, init):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        #init
        if init == 'zero':
            delta.zeros_like(X).cuda()
        elif init == 'random':
            if norm == 'l2-scaled':
                delta = torch.zeros_like(X).cuda().normal_()
                dnorm = delta.view(delta.size(0),-1).norm(p=2,dim=1).view(delta.size(0),1,1,1)
                r = torch.zeros_like(dnorm).uniform_(0, 1)
                delta.data *=  r * epsilon / dnorm 
            else:
                delta = torch.zeros_like(X).cuda().uniform_(-epsilon, epsilon)
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
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]

            if norm == 'linf':
                d = torch.clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            elif norm == 'l2': #using sign of gradient --> limits directions of gradient
                d = d + alpha * torch.sign(g)
                d_flat = d.view(d.size(0),-1)
                norm = d_flat.norm(p=2,dim=1).clamp(min=epsilon).view(d.size(0),1,1,1)
                d *=  epsilon / norm
            elif norm == 'l2-scaled':
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)            

            d = clamp(d, 0-x, 1-x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta