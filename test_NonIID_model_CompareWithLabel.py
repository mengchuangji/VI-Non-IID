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
import scipy.io as io


import segyio
from datasets.gain import *
from networks import VDN
from utils import load_state_dict_cpu, peaks, sincos_kernel, generate_gauss_kernel_mix, batch_SSIM
import math
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim

def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)



def save_result(result, path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))
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
def show(x,y,x_,n,x_max,sigma2):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,3))
    plt.subplot(171)
    plt.imshow(y,vmin=-1,vmax=1,cmap=plt.cm.seismic) #plt.cm.seismic 'gray'
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('noised')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(172)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x,vmin=-1,vmax=1,cmap=plt.cm.seismic)
    plt.title('label')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(173)
    # x_ = gain(x_, 0.004, 'agc', 0.05, 1)
    plt.imshow(x_,vmin=-1,vmax=1,cmap=plt.cm.seismic)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('denoised')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(174)
    noise= y-x_
    # residual = gain(residual, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(noise,vmin=-1,vmax=1,cmap=plt.cm.seismic)
    plt.title('noise')
    # plt.colorbar(shrink=0.5)

    plt.subplot(175)
    n_real=n
    n_=y - x
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y-x,vmin=-1,vmax=1,cmap=plt.cm.seismic)
    plt.title('groundtruth noise')
    # plt.colorbar(shrink=0.5)


    plt.subplot(176)
    residual= x_-x
    plt.imshow(residual, vmin=-1,vmax=1,cmap=plt.cm.seismic)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('residual')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(177)
    plt.imshow(np.sqrt(sigma2),cmap=plt.cm.jet)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('sigma')
    plt.colorbar(shrink= 0.8)
    plt.show()

def showsigma(sigma2):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,9))

    plt.imshow(np.sqrt(sigma2))#, vmin=0,vmax=1
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('sigma')
    plt.colorbar(shrink= 0.8)
    plt.show()
    # print("sigma.median:",np.median(np.sqrt(sigma2)))

def showm(m):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,9))

    plt.imshow(m)#, vmin=0,vmax=1
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('m')
    plt.colorbar(shrink= 0.8)
    plt.show()

def show_gain(x, y, x_, n, sigma2):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 3))
    plt.subplot(161)
    y_gain = gain(y, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y_gain, vmin=-1, vmax=1, cmap='gray')
    plt.title('noised')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(162)
    x_gain = gain(x, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x_gain, vmin=-1, vmax=1, cmap='gray')
    plt.title('original')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(163)
    x__gain = gain(x_, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x__gain, vmin=-1, vmax=1, cmap='gray')
    plt.title('denoised')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(164)
    noise = y - x_
    noise_gain = gain(noise, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(noise_gain, vmin=-1, vmax=1, cmap='gray')
    plt.title('noise')
    # io.savemat(('./noise/vdnnseming.mat'), {'data': noise_gain[:, :, np.newaxis]})
    # plt.colorbar(shrink=0.5)

    plt.subplot(165)
    n_real = n
    n_ = y - x
    n__gain = gain(n_, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(n__gain, vmin=-1, vmax=1, cmap='gray')
    plt.title('groundtruth noise')
    # io.savemat(('./noise/702nog.mat'), {'data': n__gain[:, :, np.newaxis]})
    # plt.colorbar(shrink=0.5)

    plt.subplot(166)
    residual = x_ - x
    residual_gain = gain(residual, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(residual_gain, vmin=-1, vmax=1, cmap='gray')
    plt.title('residual')
    # plt.colorbar(shrink= 0.5)
    plt.show()



def readsegy(data_dir, file):
        filename = os.path.join(data_dir, file)
        with segyio.open(filename, 'r', ignore_geometry=True) as f:
            f.mmap()
            #print binary header info
            print(f.bin)
            print(f.bin[segyio.BinField.Traces])
            # read headerword inline for trace 10
            print(f.header[10][segyio.TraceField.INLINE_3D])
            # Print inline and crossline axis
            print(f.xlines)
            print(f.ilines)

            # Extract header word for all traces
            sourceX = f.attributes(segyio.TraceField.SourceX)[:]

            trace_num = len(sourceX)  # number of trace, The sourceX under the same shot is the same character.
            data = np.asarray([np.copy(x) for x in f.trace[:trace_num]]).T
            x = data[:, :]#[76:876,0:480]
            # x = data[:, :][476:876, 0:128]
            f.close()
            return x






class DnCNN(nn.Module):

    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

use_gpu = True
case = 5
C = 1
dep_U = 4
# # clip bound
log_max = math.log(1e4)
log_min = math.log(1e-8)

if __name__ == '__main__':


    torch.set_default_dtype(torch.float32)
    # load the pretrained model
    print('Loading the Model')
    checkpoint = torch.load('./TrainedModel/Non-IID-Unet/model_state_10')
   # ./TrainedModel/Non-IID-Unet/model_state_10




    model = VDN(C, dep_U=dep_U, wf=64)
    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(checkpoint)
    else:
        load_state_dict_cpu(model, checkpoint)
    model.eval()

    snrs = []

    data_dir='test_data'
    im = '00-L120.sgy'

    original=readsegy(data_dir,'00-L120.sgy')#[0:64,192:256]#[0:64,192:256] #[64:128,364:428]#[0:128,300:428]
    groundtruth = readsegy(data_dir, '00-L120-Y.sgy')#[0:64,192:256]#[0:64,192:256] #[64:128,364:428]#[0:128,300:428]
    noise = readsegy(data_dir, '00-L120-N.sgy')#[0:64,192:256]#[0:64,192:256] #[64:128,364:428]#[0:128,300:428]


#############
    x=original
    # H, W = x.shape
    # if H % 2 ** dep_U != 0:
    #      H -= H % 2 ** dep_U
    # if W % 2 ** dep_U != 0:
    #     W -= W % 2 ** dep_U
    # x = x[:H, :W, ]
    # groundtruth=groundtruth[:H, :W, ]
    # noise=noise[:H, :W, ]
    #############################
    # x_max=x.max()
    x_max=max(abs(original.max()),abs(original.min()))
    x=x/x_max
    # io.savemat(('./noise/XJ-noisy.mat'), {'data': x[:, :]})
    groundtruth=groundtruth/x_max
    # io.savemat(('./noise/XJ-gt.mat'), {'data': x[:, :]})
    noise=noise/x_max
    # io.savemat(('./noise/702no.mat'), {'data': noise[:, :, np.newaxis]})
    ##############################
    from skimage.measure import compare_psnr
    psnr_y = compare_psnr(groundtruth, x)
    print(' psnr_y_before= {1:2.2f}dB'.format('test', psnr_y))
    snr_y = compare_SNR(groundtruth, x)
    print(' snr_y= {1:2.2f}dB'.format('test', snr_y))
    y_ssim = compare_ssim(x, y, data_range=2)
    print('ssim_before=', '{:.4f}'.format(y_ssim))
    ##################################
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
        phi_sigma = phi_sigma  # /phi_sigma.max()
        log_alpha = phi_sigma[:, :C, ]
        alpha = torch.exp(log_alpha)
        log_beta = phi_sigma[:, C:, ]
        beta = torch.exp(log_beta)

        sigma2 = beta / (alpha + 1)
        sigma2 = sigma2.cpu().numpy().squeeze()
        # io.savemat(('./noise/XJ-sigma-vdn-ul.mat'), {'data': np.squeeze(np.sqrt(sigma2))})
        sigma = np.sqrt(sigma2)
        print("sigma.min:", sigma.min(), "sigma.median:", np.median(sigma), "sigma.max:", sigma.max())

        m2 = np.exp(err[:, C:, ])  # variance
        m = np.sqrt(m2.squeeze())
        # io.savemat(('./noise/XJ-m-vdn-ul.mat'), {'data': m})
        print("m.min:", m.min(), "m.median:", np.median(m), "m.max:", m.max())

    if use_gpu:
        x_ = x_.cpu().numpy()
    else:
        x_ = x_.numpy()
    denoised = x_ - err[:, :C, ]
    denoised = denoised.squeeze()
    no = err[:, :C, ].squeeze()
    # io.savemat(('./noise/vnun-ul-n.mat'), {'data': no[:, :]})
    # io.savemat(('./noise/vunn-ul-dn.mat'), {'data': denoised[:, :]})


    io.savemat(('./output/XJ_VINonIID_n.mat'), {'data': no[:, :]})
    io.savemat(('./output/XJ_VINonIID_dn.mat'), {'data': denoised[:, :]})
    io.savemat(('./output/XJ_VINonIID_sigma2.mat'), {'data': sigma2[:, :]})
    io.savemat(('./output/XJ_VINonIID_sigma.mat'), {'data': sigma[:, :]})

    elapsed_time = time.time() - start_time
    print(' %10s : %2.4f second' % (im, elapsed_time))
    snr_x_ = compare_SNR(groundtruth, denoised)
    psnr_x_ = compare_psnr(groundtruth, denoised)
    print('psnr_y_after=', '{:.4f}'.format(psnr_x_))
    # io.savemat(('./noise/702ori.mat'), {'data': groundtruth[:, :, np.newaxis]})
    # io.savemat(('./noise/702noise.mat'), {'data': x[:, :, np.newaxis]})


    ####################################################
    x1 = groundtruth
    x_1 = denoised
    ssim = compare_ssim(x1, x_1)
    print('ssim_after=','{:.4f}'.format(ssim))

    ################################################


    if True:
        # name, ext = os.path.splitext(im)
        show(groundtruth, x, denoised,noise,x_max,sigma2)
        show_gain(groundtruth, x, denoised, noise, sigma2)
        # showsigma(sigma2)
        # showm(m)
        from datasets import wigb
        # wigb.wigb(noise)
        # showm(conf_intveral[0])
        # showm(conf_intveral[1])
        # showm(probality)

        # show_gain(groundtruth, x, denoised,noise,sigma2)#mcj


        # save_result(x_, path=os.path.join(args.result_dir, 'DnCNN' + '.png'))  # save the denoised image
    snrs.append(snr_x_)
    snr_avg = np.mean(snrs)
    snrs.append(snr_avg)
    # if args.save_result:
    #     save_result(snrs, path=os.path.join(args.result_dir,'results.txt'))
    # log('Datset: {0:10s} \n  SNR = {1:2.2f}dB'.format('test', snr_avg))








