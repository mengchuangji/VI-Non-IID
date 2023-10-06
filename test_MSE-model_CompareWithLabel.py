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
from skimage.io import imread, imsave

import segyio
from datasets.gain import *
from utils import peaks, sincos_kernel, generate_gauss_kernel_mix
import scipy.io as io
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
from skimage import img_as_float, img_as_ubyte

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/test/original', type=str, help='directory of test dataset')
    parser.add_argument('--sigma', default=50, type=float, help='noise level')
    parser.add_argument('--agc', default=False, type=bool, help='Agc operation of the data,True or False')

    # parser.add_argument('--model_dir', default=os.path.join('models_denoise', 'DnCNN_fielddata/DnCNN_real5_test'), help='directory of the model')
    # parser.add_argument('--model_dir', default=os.path.join('models_denoise', 'fielddata/simu/DnCNN'),help='directory of the model')
    parser.add_argument('--model_dir', default=os.path.join('models_denoise', '1111/fre/f_s_7/5DnMSE'))


    parser.add_argument('--model_name', default='model_050.pth', type=str, help='the model name')
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
def NormMinandMax(npdarr, min=0, max=1):
    """"
    将数据npdarr 归一化到[min,max]区间的方法
    返回 副本
    """
    arr = npdarr.flatten()
    Ymax = np.max(arr)  # 计算最大值
    Ymin = np.min(arr)  # 计算最小值
    k = (max - min) / (Ymax - Ymin)
    last = min + k * (npdarr - Ymin)

    return last
def show(x,y,x_,n,x_max):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,3))
    plt.subplot(161)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x,vmin=-1,vmax=1,cmap='gray')
    plt.title('original')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(162)
    plt.imshow(y,vmin=-1,vmax=1,cmap='gray')
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('noised')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(163)
    # x_ = gain(x_, 0.004, 'agc', 0.05, 1)
    plt.imshow(x_,vmin=-1,vmax=1,cmap='gray')
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('denoised')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(164)
    noise= y-x_
    # residual = gain(residual, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(noise,vmin=-1,vmax=1,cmap='gray')
    plt.title('noise')
    # plt.colorbar(shrink=0.5)

    plt.subplot(165)
    n_real=n
    n_=y - x
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y-x,vmin=-1,vmax=1,cmap='gray')
    plt.title('groundtruth noise')
    # plt.colorbar(shrink=0.5)


    plt.subplot(166)
    residual= x_-x
    plt.imshow(residual, vmin=-1,vmax=1,cmap='gray')
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.title('residual')
    # plt.colorbar(shrink= 0.5)
    plt.show()

def show_gain(x, y, x_, n, x_max):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 3))
    plt.subplot(161)
    x_gain = gain(x, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(x_gain, vmin=-1, vmax=1, cmap='gray')
    plt.title('original')
    # plt.colorbar(shrink= 0.5)

    plt.subplot(162)
    y_gain = gain(y, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(y_gain, vmin=-1, vmax=1, cmap='gray')
    plt.title('noised')
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
    # plt.colorbar(shrink=0.5)
    # io.savemat(('./noise/DnCNNseming.mat'), {'data': noise_gain[:, :, np.newaxis]})


    plt.subplot(165)
    n_real = n
    n_ = y - x
    n__gain = gain(n_, 0.004, 'agc', 0.05, 1)
    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    plt.imshow(n__gain, vmin=-1, vmax=1, cmap='gray')
    plt.title('groundtruth noise')
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
            sourceX = f.attributes(segyio.TraceField.SourceX)[:]
            trace_num = len(sourceX)  # number of trace, The sourceX under the same shot is the same character.
            if trace_num > 500:
                data = np.asarray([np.copy(x) for x in f.trace[0:500]]).T

                if data.shape[0] > 600:
                    x = data[400:527, 100:227]
                else:
                    x = data[:, :]
            else:
                data = np.asarray([np.copy(x) for x in f.trace[:]]).T

                if data.shape[0] > 600:

                    # x = data[100:420, 100:420]
                    # x = data[300:620, 100:420]
                    # x = data[500:820, 100:420]
                    x = data[76:876, 0:480]
                    # x = data[0:876, 0:500]
                    # x = gain(x, 0.004, 'agc', 0.05, 1)
                    x = x.astype(np.float32)
                else:
                    x = data[:, :]
            f.close()
            return x

# class DnCNN(nn.Module):
#
#     def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
#         super(DnCNN, self).__init__()
#         kernel_size = 2
#         padding = 1
#         layers = []
#         layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
#         layers.append(nn.ReLU(inplace=True))
#         for _ in range(depth-2):
#             layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
#             layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
#             layers.append(nn.ReLU(inplace=True))
#         layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
#         self.dncnn = nn.Sequential(*layers)
#         self._initialize_weights()
#
#     def forward(self, x):
#         y = x
#         out = self.dncnn(x)
#         return y-out
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.orthogonal_(m.weight)
#                 print('init weight')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)

case = 3
if __name__ == '__main__':

    args = parse_args()
    log('load trained model')
    # MSE-DnCNN
    # from networks.residual import DnCNN_Residual
    # model=DnCNN_Residual()
    # model = torch.load('./TrainedModel/MSE-DnCNN/model_050.pth')

    # MSE-UNet
    from networks.UNet import UNet
    model = UNet(in_channels=1, out_channels=1, depth=4, wf=64, slope=0.2)
    model = torch.load('./TrainedModel/MSE-Unet/model_050.pth')


    torch.set_default_dtype(torch.float32)





    # model=model.load_state_dict(model['model'])
    model.eval()  # evaluation mode

    if torch.cuda.is_available():
        model = model.cuda()

    snrs = []

    data_dir='test_data'
    im = '00-L120.sgy'
    original=readsegy(data_dir,'00-L120.sgy')#[0:64,192:256]
    groundtruth = readsegy(data_dir, '00-L120-Y.sgy')#[0:64,192:256]
    noise = readsegy(data_dir, '00-L120-N.sgy')#[0:64,192:256]

    ###########
    np.random.seed(seed=0)  # for reproducibility
    y=original
    #################################
    x=groundtruth

    ##################################
    y_max=max(abs(original.max()),abs(original.min()))
    y=y/y_max
    x=x/y_max
    noise=noise/y_max
    ##################################
    snr_y = compare_SNR(x, y)
    print(' snr_y= {1:2.2f}dB'.format('test', snr_y))
    psnr_y = compare_psnr(x, y)
    print(' psnr_y_before= {1:2.2f}dB'.format('test', psnr_y))
    y_ssim = compare_ssim(x, y, data_range=2)
    print('ssim_before=', '{:.4f}'.format(y_ssim))
    #####################################
    y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])
    torch.cuda.synchronize()
    start_time = time.time()
    y_ = y_.cuda()
    x_ = model(y_)  # inferences
    x_ = x_.view(y.shape[0], y.shape[1])
    x_ = x_.cpu()
    x_ = x_.detach().numpy().astype(np.float32)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    print(' %10s : %2.4f second' % (im, elapsed_time))
    snr_x_ = compare_SNR(x, x_)
    psnr_x_ = compare_psnr(x, x_)
    print('psnr_x_after=', '{:.4f}'.format(psnr_x_))

    no=y_.squeeze().cpu().detach().numpy().astype(np.float32)-x_
    # io.savemat(('./noise/dn-ul-n.mat'), {'data': no[:, :, np.newaxis]})
    # io.savemat(('./noise/dn-ul-dn.mat'), {'data': x_[:, :, np.newaxis]})
    # io.savemat(('./noise/ma_denoise_75_cese3.mat'), {'data': x_})
    ####################################################



    x_1 = img_as_ubyte(x_.clip(-1, 1))
    ssim = compare_ssim(x1, x_1, data_range=255, gaussian_weights=True,
                        use_sample_covariance=False, multichannel=False)
    print('ssim_after=', '{:.4f}'.format(ssim))
    ################################################



    if args.save_result:
        name, ext = os.path.splitext(im)
        show(x, y, x_,noise,y_max)
        show_gain(x, y, x_, noise, y_max)
        save_result(x_, path=os.path.join(args.result_dir, name + '_dncnn' + '.png'))  # save the denoised image
    snrs.append(snr_x_)



    snr_avg = np.mean(snrs)
    snrs.append(snr_avg)
    if args.save_result:
        save_result(snrs, path=os.path.join(args.result_dir,'results.txt'))
    log('Datset: {0:10s} \n  SNR = {1:2.2f}dB'.format('test', snr_avg))








