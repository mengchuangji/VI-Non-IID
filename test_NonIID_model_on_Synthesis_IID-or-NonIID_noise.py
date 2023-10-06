# -*- coding: utf-8 -*-

import argparse
import random
import os, time, datetime
from datasets.DenoisingDatasets_seismic import gaussian_kernel
from datasets.get_patch import *
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from scipy.io import loadmat
from skimage import img_as_ubyte
from skimage.io import imread, imsave
import scipy.io as io
from networks import VDN
import segyio
from datasets.gain import *
from utils import load_state_dict_cpu, peaks, sincos_kernel, generate_gauss_kernel_mix, batch_SSIM
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/test', type=str, help='directory of test dataset')
    parser.add_argument('--sigma', default=25, type=float, help='noise level')
    parser.add_argument('--agc', default=False, type=bool, help='Agc operation of the data,True or False')
    parser.add_argument('--model_dir', default=os.path.join('models_denoise', 'DnCNN_sigma50'), help='directory of the model')
    parser.add_argument('--model_name', default='model.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results_denoise', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()


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

def show(x,y,x_):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16,3))
    plt.subplot(161)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x,vmin=-1,vmax=1,cmap='gray')
    plt.title('original')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(162)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y,vmin=-1,vmax=1,cmap='gray')
    plt.title('noised')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(163)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x_,vmin=-1,vmax=1,cmap='gray')
    plt.title('denoised')
    # plt.colorbar(shrink= 0.5)


    plt.subplot(164)
    noise=y-x_
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(noise, vmin=-1, vmax=1,cmap='gray')
    plt.title('noise')
    # plt.colorbar(shrink=0.5)

    plt.subplot(165)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y-x, vmin=-1, vmax=1,cmap='gray')
    plt.title('groundtruth noise')
    # plt.colorbar(shrink=0.5)

    plt.subplot(166)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x_-x, vmin=-1, vmax=1,cmap='gray')
    plt.title('residual')
    # plt.colorbar(shrink=0.5)
    plt.show()

def showsigma(sigma2):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,9))

    plt.imshow(sigma2)# vmin=0,vmax=1
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('sigma2')
    plt.colorbar(shrink= 0.8)
    plt.show()


def generate_sigma():
        pch_size = 128
        # center = [random.uniform(0, pch_size), random.uniform(0, pch_size)]
        center = [pch_size/2, pch_size/2]
        # scale = random.uniform(pch_size/4*1, pch_size/4*3)
        scale = pch_size/4*2
        kernel = gaussian_kernel(pch_size, pch_size, center, scale)
        up = random.uniform(10/255.0, 75/255.0)
        down = random.uniform(10/255.0, 75/255.0)
        if up < down:
            up, down = down, up
        up += 5/255.0
        sigma_map = down + (kernel-kernel.min())/(kernel.max()-kernel.min())  *(up-down)
        sigma_map = sigma_map.astype(np.float32)

        return sigma_map[:, :]
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
def MonoPao():
    zz = loadmat("./datasets/seismic/NoiseLevelMap/MonoPaoSigma.mat")['data']
    zz = np.sqrt(zz)
    # print(np.mean(zz))
    print("MonoPao.median",np.median(zz))
    print("MonoPao.max",zz.max())
    print("MonoPao.min",zz.min())
    return zz

def Gauss(sigma):
    sigma0=sigma/255
    sigma1=np.ones((256,256))*sigma0
    return sigma1

def Panke100_228_19_147Sigma():
    zz = loadmat('./datasets/seismic/NoiseLevelMap/PankeSigma100_228_19_147.mat')['data']
    zz = np.sqrt(zz)
    print("Panke100_228_19_147",np.median(zz))
    print("Panke100_228_19_147",zz.max())
    print("Panke100_228_19_147",zz.min())
    return zz
def gaussian_kernel():
    H = 128
    W = 128
    center = [64, 64]
    scale = 32
    centerH = center[0]
    centerW = center[1]
    XX, YY = np.meshgrid(np.arange(W), np.arange(H))
    ZZ = 1./(2*np.pi*scale**2) * np.exp( (-(XX-centerH)**2-(YY-centerW)**2)/(2*scale**2) )
    return  ZZ

###################################################################################################

use_gpu = True
case = 2  # choose a case of Non-IID noise
C = 1
dep_U = 4
# # clip bound
log_max = math.log(1e4)
log_min = math.log(1e-8)
import math
if __name__ == '__main__':

    args = parse_args()
    torch.set_default_dtype(torch.float64)
    # load the pretrained model
    print('Loading the Model')

    #model:   This model is trained on Marmousi model. We use the Marmousi model to calculate the reflection
    # coefficients, then we use reflection coefficients and the Ricker
    # wavelet with a dominant frequency of 35Hz and phase of 0 to synthesize the clean data of the training set
    checkpoint = torch.load('./TrainedModel/simu-Non-IID-Unet/model_state_26')
    # torch.save(checkpoint, './TrainedModel/simu-Non-IID-Unet/model_state_26_new', _use_new_zipfile_serialization=False)

    model = VDN(C, dep_U=dep_U, wf=64)
    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(checkpoint)
    else:
        load_state_dict_cpu(model, checkpoint)
    model.eval()

    snrs = []


    #########################################
    im = 'overthrust_204'
    data_dir = './test_data/overthrust/'
    original = io.loadmat(data_dir + 'overthrust_204.mat')['data']#.astype(np.float32) # 187*801

    x=original[0:160, 320:448]#[0:160, 320:480] #[1000:1640,7000:7640] #[0:160, 320:480]
    # x = x / x_max

    H, W = x.shape
    if H % 2 ** dep_U != 0:
        H -= H % 2 ** dep_U
    if W % 2 ** dep_U != 0:
        W -= W % 2 ** dep_U
    x = x[:H, :W, ]

    # Generate the sigma map
    if case == 1:
        # Test case 1
        sigma = peaks(256)
    elif case == 2:
        # Test case 2
        sigma = sincos_kernel()
    elif case == 3:
        # Test case 3
        sigma = generate_gauss_kernel_mix(256, 256)
    elif case == 4:
        sigma = Panke100_228_19_147Sigma()
    elif case == 5:
        sigma = MonoPao()
    elif case == 6:
        sigma = gaussian_kernel()
    else:
        sys.exit('Please input the corrected test case: 1, 2 or 3')
    sigma = 10/ 255.0 + (sigma - sigma.min()) / (sigma.max() - sigma.min()) * ((255*0.8-10) / 255.0)
    sigma = cv2.resize(sigma, (W, H))

    # # ###########
    np.random.seed(seed=0)  # for reproducibility
    # # #######################
    x_max = max(abs(x.max()), abs(x.min()))

    ####################### add Non-IID gauss nosie #####################
    y = x + np.random.normal(0, 1, x.shape)* sigma[:, :]#*x_max
    ####################### add IID gauss nosie #####################
    # y = x + np.random.normal(0, 75 / 255.0, x.shape)*x_max

    # ################################
    snr_y = compare_SNR(x, y)
    print(' snr_y= {1:2.2f}dB'.format('test', snr_y))
    psnr_y = compare_psnr(x, y)
    print('psnr_y_before=', '{:.4f}'.format(psnr_y))
    y_ssim = compare_ssim(x, y, data_range=2)
    print('ssim_before=', '{:.4f}'.format(y_ssim))
    ##################################
    # y=loadmat('./test_data/seismic/pao1.mat')['d'][:, :].clip(-50, 50)[1000:1128,50:178 ]
    # y=y/y.max()
    ################################
    y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])
    torch.cuda.synchronize()
    start_time = time.time()
    if use_gpu:
        y_ = y_.cuda()
        print('Begin Testing on GPU')
    else:
        print('Begin Testing on CPU')
    # Variable description
    # the output of NA-Net: phi_Z ({\mu}^hat), m2 (sigma^{hat})
    # the output of NLE-Net: alpha (\lambda),  beta (\zeta)
    # The characters in brackets correspond to the characters in our paper.
    with torch.autograd.set_grad_enabled(False):
        from seis_utils.PadUnet import PadUnet
        padunet = PadUnet(y_, dep_U=5)
        y_pad = padunet.pad()
        phi_Z_pad = model(y_pad, 'test')
        phi_Z = padunet.pad_inverse(phi_Z_pad)
        err = phi_Z.cpu().numpy()
        phi_sigma_pad = model(y_pad, 'sigma')
        phi_sigma = padunet.pad_inverse(phi_sigma_pad)
        phi_sigma.clamp_(min=log_min, max=log_max)
        phi_sigma = phi_sigma  # /phi_sigma.max()
        log_alpha = phi_sigma[:, :C, ]
        alpha = torch.exp(log_alpha)
        log_beta = phi_sigma[:, C:, ]
        beta = torch.exp(log_beta)

        sigma2 = beta / (alpha + 1)
        sigma2 = sigma2.cpu().numpy()
        # io.savemat(('./noise/GaussSigma.mat'), {'data': np.squeeze(sigma2)})
        sigma = np.sqrt(sigma2).squeeze()
        from datasets.data_tools import sigma_estimate

        print("sigma2.min:", sigma2.min(), "sigma2.median:", np.median(sigma), "sigma2.ave:", np.average(sigma),
              "sigma2.max:", sigma.max())
        m2 = np.exp(err[:, C:, ])  # variance
        m = np.sqrt(m2.squeeze())
        print("m.min:", m.min(), "m.median:", np.median(m), "m.max:", m.max())
    if use_gpu:
        y_ = y_.cpu().numpy()
    else:
        y_ = y_.numpy()
    x_ = y_ - err[:, :C, ]
    x_ = x_.squeeze()
    sigma2 = sigma2.squeeze()
    elapsed_time = time.time() - start_time
    print(' %10s : %2.4f second' % (im, elapsed_time))
    # #########
    # im_noisy_img = img_as_ubyte(NormMinandMax(x))
    # im_gt_img = img_as_ubyte(NormMinandMax(x_))
    # SSIM=compare_ssim(x, x_, data_range=255, multichannel=False)
    # print(SSIM)
    # ####################
    snr_x_ = compare_SNR(x, x_)
    psnr_x_ = compare_psnr(x, x_)
    print('psnr_x_after=', '{:.4f}'.format(psnr_x_))
    ####################################################
    ssim = compare_ssim(x, x_, data_range=2)
    print('ssim=', '{:.4f}'.format(ssim))
    ################################################
    # y_max = max(abs(y.copy().max()), abs(y.copy().min()))
    # x_max=y_max
    if args.save_result:
        name, ext = os.path.splitext(im)
        show(x, y, x_)
        # showsigma(sigma)
        # save_result(x_, path=os.path.join(args.result_dir, name + '_dncnn' + '.png'))  # save the denoised image
        from datasets import wigb
        # wigb.wigb(x.copy())
        # wigb.wigb(y.copy() )
        # wigb.wigb(x_.copy())
        # # wigb.wigb(y.copy() - x.copy())
        # wigb.wigb(y.copy()-x_.copy())
        # # wigb.wigb(x-x_)

        # x_max=1
        # 设置横纵坐标的名称以及对应字体格式
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 30,
                 }

        wigb.wigb((x.copy()) /x_max, figsize=(30, 20),no_plot=True)
        plt.xticks([])
        plt.yticks([])

        # bwith=2
        # ax = plt.gca()  # 获取边框
        # ax.spines['bottom'].set_linewidth(bwith)
        # ax.spines['left'].set_linewidth(bwith)
        # ax.spines['top'].set_linewidth(bwith)
        # ax.spines['right'].set_linewidth(bwith)
        #
        # plt.gca().xaxis.set_ticks(np.arange(0, 129, 30))
        # plt.gca().yaxis.set_ticks(np.arange(0, 161, 40))
        # plt.xlabel('Trace',font2)
        # plt.ylabel('Time sampling number',font2)
        # plt.gca().set_xticklabels(np.arange(320, 449, 30))
        # # 设置坐标刻度值的大小以及刻度值的字体
        # plt.tick_params(labelsize=26)
        # labels = plt.gca().get_xticklabels() + plt.gca().get_yticklabels()
        # [label.set_fontname('Times New Roman') for label in labels]
        # plt.xlim(320, 480)  # 0:160, 320:480
        # plt.ylim(160, 0)
        # 将文件保存至文件中并且画出图
        # plt.savefig('figure.eps')
        # plt.savefig('E:\VIRI\paper\\1stPaperSE\mcj-sencond-material\pictures\wigbOuput\clean.png', format='png', dpi=50,bbox_inches='tight')

        wigb.wigb(y.copy() / x_max, figsize=(30, 20),no_plot=False)
        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig('E:\VIRI\paper\\1stPaperSE\mcj-sencond-material\pictures\wigbOuput\second\\c2-noisy.png',
        #             format='png',
        #             dpi=50, bbox_inches='tight')

        wigb.wigb(x_.copy()/ x_max, figsize=(30, 20),no_plot=False)
        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig('E:\VIRI\paper\\1stPaperSE\mcj-sencond-material\pictures\wigbOuput\second\\vun-dn-g.png', format='png',
        #             dpi=50, bbox_inches='tight')
        wigb.wigb((y.copy() - x.copy()) / x_max, figsize=(30, 20),no_plot=False)
        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig('E:\VIRI\paper\\1stPaperSE\mcj-sencond-material\pictures\wigbOuput\second\\c2-noise.png', format='png',
        #             dpi=50, bbox_inches='tight')
        wigb.wigb((y.copy() - x_.copy()) / x_max, figsize=(30, 20),no_plot=False)
        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig('E:\VIRI\paper\\1stPaperSE\mcj-sencond-material\pictures\wigbOuput\second\\vun-n-g.png', format='png',
        #             dpi=50, bbox_inches='tight')
        wigb.wigb((x.copy() - x_.copy()) / x_max, figsize=(30, 20),no_plot=False)
        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig('E:\VIRI\paper\\1stPaperSE\mcj-sencond-material\pictures\wigbOuput\second\\vun-r-g.png', format='png',
        #             dpi=50, bbox_inches='tight')








