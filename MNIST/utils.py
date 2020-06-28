import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import laplace

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def norms_l1(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:,None,None,None]

def l1_dir_topk(grad, delta, X, k = 20):
    X_curr = X + delta
    batch_size = X.shape[0]
    channels = X.shape[1]
    pix = X.shape[2]

    grad = grad.detach().cpu().numpy()
    abs_grad = np.abs(grad)
    sign = np.sign(grad)

    max_abs_grad = np.percentile(abs_grad, k, axis=(1, 2, 3), keepdims=True)
    tied_for_max = (abs_grad >= max_abs_grad).astype(np.float32)
    num_ties = np.sum(tied_for_max, (1, 2, 3), keepdims=True)
    optimal_perturbation = sign * tied_for_max / num_ties

    optimal_perturbation = torch.from_numpy(optimal_perturbation).cuda()
    return optimal_perturbation.view(batch_size, channels, pix, pix)


def proj_l1ball(x, epsilon=12):
    assert epsilon > 0
    u = x.abs()
    if (u.sum(dim = (1,2,3)) <= epsilon).all():
        return x
    y = proj_simplex(u, s=epsilon)
    y = y.view_as(x)
    y *= x.sign()   
    return y

 
def proj_simplex(v, s=1):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    batch_size = v.shape[0]
    u = v.view(batch_size,1,-1)
    n = u.shape[2]
    u, indices = torch.sort(u, descending = True)
    cssv = u.cumsum(dim = 2)
    vec = u * torch.arange(1, n+1).float().cuda()
    comp = (vec > (cssv - s)).half()
    u = comp.cumsum(dim = 2)
    w = (comp-1).cumsum(dim = 2)
    u = u + w
    rho = torch.argmax(u, dim = 2)
    rho = rho.view(batch_size)
    c = torch.HalfTensor([cssv[i,0,rho[i]] for i in range( cssv.shape[0]) ]).cuda()
    c = c-s
    theta = torch.div(c.float(), (rho.float() + 1))
    theta = theta.view(batch_size,1,1,1)
    w = (v.float() - theta).clamp(min=0)
    return w

def attack_fgsm(model, X, y, epsilon, alpha, norm, init):
    return attack_pgd(model, X, y, epsilon, alpha, 1, 1, norm, init)

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, norm, init):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        #init
        if init == 'zero':
            delta = torch.zeros_like(X).cuda()
        elif init == 'random':
            if norm == 'l2-scaled':
                delta = torch.zeros_like(X).cuda().normal_()
                dnorm = delta.view(delta.size(0),-1).norm(p=2,dim=1).view(delta.size(0),1,1,1)
                r = torch.zeros_like(dnorm).uniform_(0, 1)
                delta.data *=  r * epsilon / dnorm
            elif norm == 'l1':
                delta = torch.zeros_like(X).cuda()
                ini = laplace.Laplace(
                    loc=delta.new_tensor(0), scale=delta.new_tensor(1))
                delta.data = ini.sample(delta.data.shape)
                delta.data = (2.0*delta.data - 1.0) * epsilon 
                delta.data /= norms_l1(delta.detach()).clamp(min=epsilon)
                delta.data = clamp(delta, 0-X, 1-X)
            else:
                delta = torch.zeros_like(X).cuda().uniform_(-epsilon, epsilon)
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
            elif norm == 'l1':
                k = 20
                d = d + alpha * l1_dir_topk(g, d, x, k)
                d = proj_l1ball(d, epsilon=epsilon)            

            d = clamp(d, 0-x, 1-x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta