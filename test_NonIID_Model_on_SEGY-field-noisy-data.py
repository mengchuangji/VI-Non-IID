# -*- coding: utf-8 -*-
from datasets.get_patch import *
import argparse
import random
import os, time, datetime
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from scipy.io import loadmat
from skimage import img_as_ubyte
from skimage.io import imread, imsave
# from skimage.measure import compare_ssim
import scipy.io as io


import segyio
from datasets.gain import *
from networks import VDN, UNet, DnCNN
from utils import load_state_dict_cpu, peaks, sincos_kernel, generate_gauss_kernel_mix, batch_SSIM
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/fielddata', type=str, help='directory of test dataset')
    parser.add_argument('--sigma', default=75, type=float, help='noise level')
    parser.add_argument('--agc', default=False, type=bool, help='Agc operation of the data,True or False')
    parser.add_argument('--model_dir', default=os.path.join('models_denoise', 'DnCNN_sigma50'), help='directory of the model')
    parser.add_argument('--model_name', default='model.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results_denoise', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()



def compare_SNR(real_img,recov_img):
    real_mean = np.mean(real_img)
    tmp1 = real_img - real_mean
    real_var = sum(sum(tmp1*tmp1))

    noise = real_img - recov_img
    noise_mean = np.mean(noise)
    tmp2 = noise - noise_mean
    noise_var = sum(sum(tmp2*tmp2))

    if noise_var ==0 or real_var==0:
      s = 999.99
    else:
      s = 10*math.log(real_var/noise_var,10)
    return s
def show(x,y,sigma2,x_max,figsize):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.subplot(141)
    plt.imshow(x,vmin=-1,vmax=1,cmap='gray')
    # plt.xticks([])  # 去掉横坐标值
    # plt.yticks([])  # 去掉纵坐标值
    plt.title('original')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(142)
    # plt.xticks([])  # 去掉横坐标值
    # plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y,vmin=-1,vmax=1,cmap='gray')
    plt.title('denoised')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(143)
    noise= x-y
    # residual = gain(residual, 0.004, 'agc', 0.05, 1)
    # plt.xticks([])  # 去掉横坐标值
    # plt.yticks([])  # 去掉纵坐标值
    plt.imshow(noise,vmin=-1,vmax=1,cmap='gray')
    plt.title('noise')


    plt.subplot(144)
    # x_ = gain(x_, 0.004, 'agc', 0.05, 1)
    plt.imshow(sigma2)
    # plt.xticks([])  # 去掉横坐标值
    # plt.yticks([])  # 去掉纵坐标值
    plt.title('sigma2')
    plt.colorbar(shrink=0.5)
    plt.show()


def readsegy(data_dir, file,j):
        filename = os.path.join(data_dir, file)
        with segyio.open(filename, 'r', ignore_geometry=True) as f:
            f.mmap()
            sourceX = f.attributes(segyio.TraceField.SourceX)[:]
            trace_num = len(sourceX)  # number of trace, The sourceX under the same shot is the same character.
            shot_num = int(float(trace_num / 224))# 224 787
            len_shot = trace_num // shot_num  # The length of the data in each shot data
            data = np.asarray([np.copy(x) for x in f.trace[j * len_shot:(j + 1) * len_shot]]).T
            # data = data/data.max()
            # data = data  # Not normalized
            x = data[:, :]
            f.close()
            return x



def readsegy_all(data_dir, file):
    filename = os.path.join(data_dir, file)
    with segyio.open(filename, 'r', ignore_geometry=True) as f:
        f.mmap()
        sourceX = f.attributes(segyio.TraceField.SourceX)[:]
        trace_num = len(sourceX)  # number of trace, The sourceX under the same shot is the same character.
        data = np.asarray([np.copy(x) for x in f.trace[:trace_num]]).T
        f.close()
        return data



class VDN(nn.Module):
    def __init__(self, in_channels, wf=64, dep_S=5, dep_U=4, slope=0.2):
        super(VDN, self).__init__()
        # #VDN Unet
        self.DNet = UNet(in_channels, in_channels*2, wf=wf, depth=dep_U, slope=slope)
        # self.DNet = DnCNN(in_channels, in_channels * 2, dep=17, num_filters=64, slope=slope)
        # self.DNet = DnCNN_R(in_channels, in_channels*2, dep=17, num_filters=64, slope=slope)
        self.SNet = DnCNN(in_channels, in_channels*2, dep=dep_S, num_filters=64, slope=slope)

    def forward(self, x, mode='train'):
        if mode.lower() == 'train':
            phi_Z = self.DNet(x)
            phi_sigma = self.SNet(x)
            return phi_Z, phi_sigma
        elif mode.lower() == 'test':
            phi_Z = self.DNet(x)
            return phi_Z
        elif mode.lower() == 'sigma':
            phi_sigma = self.SNet(x)
            return phi_sigma

use_gpu = True
C = 1
dep_U = 4
# # clip bound
log_max = math.log(1e4)
log_min = math.log(1e-8)

if __name__ == '__main__':

    args = parse_args()
    torch.set_default_dtype(torch.float32)
    # load the pretrained model
    print('Loading the Model')

    # VI-non-IID Unet
    checkpoint = torch.load('./TrainedModel/Non-IID-Unet/model_state_10')
    # VI-non-IID DnCNN
    # checkpoint = torch.load('./TrainedModel/Non-IID-DnCNN/model_state_15')

    model = VDN(C, dep_U=dep_U, wf=64)

    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(checkpoint)
    else:
        load_state_dict_cpu(model, checkpoint)

    model.eval()

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir, set_cur)

    #
    data_dir = '/home/shendi_mcj/datasets/seismic/01_Louis&Arkansas&Texas/Bad'
    im = 'B-RODESSA-77-02'
    original = readsegy_all(data_dir, 'B-RODESSA-77-02.sgy')
    # original=original[:1000, :77]

    x = original

    H, W = x.shape
    if H % 2 ** dep_U != 0:
         H -= H % 2 ** dep_U
    if W % 2 ** dep_U != 0:
        W -= W % 2 ** dep_U
    x = x[:H, :W, ]

    #############################
    x_max=abs(x).max()
    x=x/x_max
    x_ = torch.from_numpy(x).view(1, -1, x.shape[0], x.shape[1])
    torch.cuda.synchronize()
    start_time = time.time()
    if use_gpu:
        x_ = x_.cuda()
        print('Begin Testing on GPU')
    else:
        print('Begin Testing on CPU')
    # Variable description
    # the output of NA-Net: phi_Z ({\mu}^hat), m2 (sigma^{hat})
    # the output of NLE-Net: alpha (\lambda),  beta (\zeta)
    # The characters in brackets correspond to the characters in our paper.
    with torch.autograd.set_grad_enabled(False):
        from seis_utils.PadUnet import PadUnet
        padunet = PadUnet(x_, dep_U=5)
        x_pad = padunet.pad()
        # Calculate phi_Z
        phi_Z_pad = model(x_pad, 'test')
        phi_Z = padunet.pad_inverse(phi_Z_pad)
        err = phi_Z.cpu().numpy()
        # Calculate sigma2
        phi_sigma_pad = model(x_pad, 'sigma')
        phi_sigma = padunet.pad_inverse(phi_sigma_pad)
        phi_sigma.clamp_(min=log_min, max=log_max)
        phi_sigma=phi_sigma#/phi_sigma.max()
        log_alpha = phi_sigma[:, :C, ]
        alpha = torch.exp(log_alpha)
        log_beta = phi_sigma[:, C:, ]
        beta = torch.exp(log_beta)
        sigma2 = beta / (alpha + 1)
        sigma2 = sigma2.cpu().numpy().squeeze()
        # io.savemat(('./noise/PK-sigma-vdn-l.mat'), {'data': np.squeeze(np.sqrt(sigma2))})
        sigma=np.sqrt(sigma2)
        print("sigma.min:",sigma.min(),"sigma.median:",np.median(sigma),"sigma.max:",sigma.max())
    if use_gpu:
        x_ = x_.cpu().numpy()
    else:
        x_ = x_.numpy()
    # Calculate denoised image
    denoised = x_ - err[:, :C, ] # err represents the residual (noise) of the output, which makes training easier.
    # Calculate sigma^{hat}
    m2=np.exp(err[:, C:,])  # variance
    m=np.sqrt(m2.squeeze())
    print("m.min:", m.min(), "m.median:", np.median(m), "m.max:", m.max())

    denoised = denoised.squeeze()
    # sigma2 = sigma2.squeeze()
    elapsed_time = time.time() - start_time
    print(' %10s : %2.4f second' % (im, elapsed_time))

    show(x,denoised,np.sqrt(sigma2),x_max,figsize=(12,20))
    print('done!')








