#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-03-20 19:48:14

import torch
import torch.nn.functional as F
from math import pi, log
from seis_utils import LogGamma
from options import set_opts
import numpy as np

args = set_opts()
log_gamma = LogGamma.apply

# clip bound
log_max = log(1e4)
log_min = log(1e-8)

def loss_fn(out_denoise, out_sigma, im_noisy, im_gt, sigmaMap, eps2, radius=3):
    '''
    Input:
        radius: radius for guided filter in the Inverse Gamma prior
        eps2: variance of the Gaussian prior of Z
    '''
    C = im_gt.shape[1]
    p = 2*radius+1
    p2 = p**2
    alpha0 = 0.5 * torch.tensor([p2-2]).type(sigmaMap.dtype).to(device=sigmaMap.device)
    beta0 = 0.5 * p2 * sigmaMap

    # parameters predicted of Gaussain distribution
    out_denoise[:, C:,].clamp_(min=log_min, max=log_max)
    err_mean = out_denoise[:, :C,]
    m2 = torch.exp(out_denoise[:, C:,])   # variance

    # parameters predicted of Inverse Gamma distribution
    out_sigma.clamp_(min=log_min, max=log_max)
    log_alpha = out_sigma[:, :C,]
    alpha = torch.exp(log_alpha)
    log_beta = out_sigma[:, C:,]
    alpha_div_beta = torch.exp(log_alpha - log_beta)

    # KL divergence for Gauss distribution
    m2_div_eps = torch.div(m2, eps2)
    err_mean_gt = im_noisy - im_gt
    kl_gauss = 0.5 * torch.mean((err_mean-err_mean_gt)**2/eps2 + (m2_div_eps - 1 - torch.log(m2_div_eps)))

    # KL divergence for Inv-Gamma distribution
    kl_Igamma = torch.mean((alpha-alpha0)*torch.digamma(alpha) + (log_gamma(alpha0) - log_gamma(alpha))
                           + alpha0*(log_beta - torch.log(beta0)) + beta0 * alpha_div_beta - alpha)

    # likelihood of im_gt
    lh = 0.5 * log(2*pi) + 0.5 * torch.mean((log_beta - torch.digamma(alpha)) + \
                                                                 (err_mean**2+m2) * alpha_div_beta)

    loss = lh + args.alpha*kl_gauss + args.beta*kl_Igamma

    return loss, lh, kl_gauss, kl_Igamma

def loss_fn_mse(out_denoise, im_noisy, im_gt):
    '''
    Input:
        radius: radius for guided filter in the Inverse Gamma prior
        eps2: variance of the Gaussian prior of Z
    '''
    out_denoise[:, :,].clamp_(min=log_min, max=log_max)
    # likelihood of im_gt
    import torch.nn as nn
    criterion = nn.MSELoss(reduce=True,size_average=False)
    lh=criterion(im_noisy-out_denoise,im_gt)
    kl_gauss = 0
    kl_Igamma = 0
    loss = lh + args.alpha*kl_gauss + args.beta*kl_Igamma
    return loss, lh, kl_gauss, kl_Igamma

def loss_fn_vi(out_denoise,im_noisy, im_gt, eps2, sigma0=75):
    '''
    Input:
        radius: radius for guided filter in the Inverse Gamma prior
        eps2: variance of the Gaussian prior of Z
    '''
    C = im_gt.shape[1]
    # parameters predicted of Gaussain distribution
    out_denoise[:, C:,].clamp_(min=log_min, max=log_max)
    err_mean = out_denoise[:, :C,]
    m2 = torch.exp(out_denoise[:, C:,])   # variance

    log_sigma0_square= np.log((sigma0/255)*(sigma0/255))

    # KL divergence for Gauss distribution
    m2_div_eps = torch.div(m2, eps2)
    err_mean_gt = im_noisy - im_gt
    kl_gauss = 0.5 * torch.mean((err_mean-err_mean_gt)**2/eps2 + (m2_div_eps - 1 - torch.log(m2_div_eps)))

    # likelihood of im_gt
    lh = 0.5 * log(2*pi) + 0.5 *log_sigma0_square + 0.5 * torch.mean(err_mean**2+m2)

    loss = lh + args.alpha*kl_gauss

    return loss, lh, kl_gauss

def loss_fn_sigma(out_denoise, out_sigma, im_noisy, im_gt, sigmaMap, eps2, radius=3):
    '''
    Input:
        radius: radius for guided filter in the Inverse Gamma prior
        eps2: variance of the Gaussian prior of Z
    '''
    C = im_gt.shape[1]
    p = 2*radius+1
    p2 = p**2
    alpha0 = 0.5 * torch.tensor([p2-2]).type(sigmaMap.dtype).to(device=sigmaMap.device)
    beta0 = 0.5 * p2 * sigmaMap

    # parameters predicted of Gaussain distribution
    out_denoise[:, C:,].clamp_(min=log_min, max=log_max)
    err_mean = out_denoise[:, :C,]
    m2 = torch.exp(out_denoise[:, C:,])   # variance

    # parameters predicted of Inverse Gamma distribution
    out_sigma.clamp_(min=log_min, max=log_max)
    log_alpha = out_sigma[:, :C,]
    alpha = torch.exp(log_alpha)
    log_beta = out_sigma[:, C:,]
    alpha_div_beta = torch.exp(log_alpha - log_beta)

    # KL divergence for Gauss distribution
    kl_gauss = 0

    # KL divergence for Inv-Gamma distribution
    kl_Igamma = torch.mean((alpha-alpha0)*torch.digamma(alpha) + (log_gamma(alpha0) - log_gamma(alpha))
                           + alpha0*(log_beta - torch.log(beta0)) + beta0 * alpha_div_beta - alpha)

    # likelihood of im_gt
    lh = 0.5 * log(2*pi) + 0.5 * torch.mean((log_beta - torch.digamma(alpha)) + \
                                                                 (err_mean**2+m2) * alpha_div_beta)

    loss = lh + args.alpha*kl_gauss + args.beta*kl_Igamma

    return loss, lh, kl_gauss, kl_Igamma

