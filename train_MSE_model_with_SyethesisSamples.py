# -*- coding: utf-8 -*-
from datasets import DenoisingDatasets_seismic
import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
# from get_patch_513 import *
from pathlib import Path
from math import ceil
import torch.nn.functional as F
from seis_utils import batch_PSNR, batch_SSIM
from networks.residual import DnCNN_Residual, UNet_Residual, NestedUNet_4_Residual


# Params
from networks import weight_init_kaiming, VDN

_C = 1

parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model', default='Unet', type=str, help='choose a type of model')
parser.add_argument('--data_dir', default='data/test', type=str, help='path of train data')
parser.add_argument('--sigma', default=75, type=int, help='noise level')
parser.add_argument('--epoch', default=50, type=int, help='number of train epoches')
# parser.add_argument('--lr', default=2e-4, type=float, help='initial learning rate for Adam')
# --lr 2e-4
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--patch_size', default=(32, 32), type=int, help='patch size')
parser.add_argument('--stride', default=(32, 32), type=int, help='the step size to slide on the data')
parser.add_argument('--jump', default=3, type=int, help='the space between shot')
parser.add_argument('--download', default=False, type=bool, help='if you will download the dataset from the internet')
parser.add_argument('--datasets', default=0, type=int,
                    help='the num of datasets you want be download,if download = True')
parser.add_argument('--train_data_num', default=100000, type=int, help='the num of the train_data')
parser.add_argument('--aug_times', default=0, type=int, help='Number of aug operations')
parser.add_argument('--scales', default=[1], type=list, help='data scaling')
parser.add_argument('--agc', default=True, type=int, help='Normalize each trace by amplitude')
parser.add_argument('--verbose', default=True, type=int, help='Whether to output the progress of data generation')
parser.add_argument('--display', default=1000, type=int, help='interval for displaying loss')

# model and log saving
parser.add_argument('--log_dir', default='./log', type=str, metavar='PATH',
                    help="Path to save the log file, (default: ./log)")
parser.add_argument('--model_dir', default='./model', type=str, metavar='PATH',
                    help="Path to save the model file, (default: ./model)")
parser.add_argument('--num_workers', default=8, type=int,
                    help="Number of workers to load data, (default: 8)")
parser.add_argument('--radius', default=3, type=int,
                    help="Radius for the Gaussian filter, (default: 3)")
parser.add_argument('-p', '--print_freq', type=int, default=500,
                    help="Print frequence (default: 100)")
parser.add_argument('-s', '--save_model_freq', type=int, default=1,
                    help="Save moel frequence (default: 1)")
parser.add_argument('--lr', type=float, default=2e-4,
                                                  help="Initialized learning rate, (default: 2e-4)")
parser.add_argument('--gamma', type=float, default=0.5,
                                         help="Decaying rate for the learning rate, (default: 0.5)")
parser.add_argument('--clip_grad_D', type=float, default=1e4,
                    help="Cliping the gradients for D-Net, (default: 1e4)")

args = parser.parse_args()


batch_size = args.batch_size
cuda = torch.cuda.is_available()
# torch.set_default_dtype(torch.float64)

n_epoch = args.epoch
sigma = args.sigma

if not os.path.exists('models_denoise'):
    os.mkdir('models_denoise')

save_dir = os.path.join('models_denoise', args.model + '_' + '11053' + str(sigma))

if not os.path.exists(save_dir):
    os.mkdir(save_dir)






def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


if __name__ == '__main__':
    # model selection
    print('===> Building model')
    torch.set_default_dtype(torch.float32)
    # model = DnCNN_Residual()
    model = UNet_Residual(in_channels=1, out_channels=1, depth=4, wf=64, slope=0.2)
    # from networks.UNet import UNet
    # model = UNet(in_channels=1, out_channels=1, depth=4, wf=64, slope=0.2)
    # model = NestedUNet_4_Residual(num_classes=1, input_channels=1, out_channels=1, deep_supervision=False, slope=0.2)

    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 1:
        print('resuming by loading epoch %03d\n' % (initial_epoch - 1))
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        if initial_epoch >= n_epoch:
            print("training have finished")

        else:
            model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % (initial_epoch - 1)))
    model.train()
    criterion = nn.MSELoss(reduce=True, size_average=False)

    if cuda:
        model = model.cuda()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    args.milestones = [10, 20, 25, 30, 35, 40, 45, 50]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, args.gamma)


    # xs = datagenerator(data_dir=args.data_dir, patch_size=args.patch_size, stride=args.stride,
    #                    train_data_num=args.train_data_num,
    #                    download=args.download, datasets=args.datasets, aug_times=args.aug_times, scales=args.scales,
    #                    verbose=args.verbose, jump=args.jump, agc=args.agc)
    # xs = xs.astype(np.float64)
    # xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))

    # making traing data
    from datasets.prepare_data.mat.bia2small_mat import generate_patch_from_mat

    train_im_list = generate_patch_from_mat(dir="/home/shendi_mcj/datasets/seismic/marmousi", pch_size=32, stride=[24,24])
    train_im_list = train_im_list.astype(np.float32)


    test_im_list = generate_patch_from_mat(dir="/home/shendi_mcj/datasets/seismic/overthrust", pch_size=128, stride=[24,64])
    test_im_list = test_im_list.astype(np.float32)

    # making iid tesing data
    # test_im_list = torch.from_numpy(test_im_list.transpose((0, 3, 1, 2)))
    # datasets = {'train':DenoisingDatasets_seismic_513.SimulateTrain(train_im_list, 5000*args.batch_size,
    #                                       args.patch_size, radius=args.radius, noise_estimate=True),
    #                      'test_cbsd681_agwn':DenoisingDataset(test_im_list,50),
    #                     'test_cbsd682_agwn': DenoisingDataset(test_im_list,75),
    #                     'test_cbsd683_agwn': DenoisingDataset(test_im_list,100)}

    # making non_iid tesing data
    test_case1_h5 = Path('/home/shendi_mcj/datasets/seismic/overthrust').joinpath('noise_niid',
                                                                                'overthrust_niid_case1.hdf5')
    test_case2_h5 = Path('/home/shendi_mcj/datasets/seismic/overthrust').joinpath('noise_niid',
                                                                                'overthrust_niid_case2.hdf5')
    test_case3_h5 = Path('/home/shendi_mcj/datasets/seismic/overthrust').joinpath('noise_niid',
                                                                                'overthrust_niid_case3.hdf5')
    test_case4_h5 = Path('/home/shendi_mcj/datasets/seismic/overthrust').joinpath('noise_niid',
                                                                                  'overthrust_niid_case4.hdf5')
    test_case5_h5 = Path('/home/shendi_mcj/datasets/seismic/overthrust').joinpath('noise_niid',
                                                                                  'overthrust_niid_case5.hdf5')


    datasets = {'train': DenoisingDatasets_seismic.DenoisingDatasets_seismic.SimulateTrain(train_im_list, 5000 * args.batch_size,
                                                       args.patch_size, radius=args.radius,
                                                         noise_estimate=True),
                'test_case1': DenoisingDatasets_seismic.SimulateTest(test_im_list, test_case1_h5),
                'test_case2': DenoisingDatasets_seismic.SimulateTest(test_im_list, test_case2_h5),
                'test_case3': DenoisingDatasets_seismic.SimulateTest(test_im_list, test_case3_h5),
                'test_case4': DenoisingDatasets_seismic.SimulateTest(test_im_list, test_case4_h5),
                'test_case5': DenoisingDatasets_seismic.SimulateTest(test_im_list, test_case5_h5)}

    _modes = ['train', 'test_case1', 'test_case2', 'test_case3','test_case4','test_case5']
    batch_size = {'train': args.batch_size, 'test_case1': 1, 'test_case2': 1, 'test_case3': 1, 'test_case4': 1,'test_case5': 1}
    num_data = {phase: len(datasets[phase]) for phase in datasets.keys()}
    num_iter_epoch = {phase: ceil(num_data[phase] / batch_size[phase]) for phase in datasets.keys()}
    param_D = [x for name, x in model.named_parameters()]
    clip_grad_D = args.clip_grad_D
    for epoch in range(initial_epoch, n_epoch):
        scheduler.step(epoch)  # step to the learning rate in this epcoh
        # DDataset = DenoisingDataset(xs, sigma)
        # DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        loss_per_epoch = {x: 0 for x in ['Loss']}
        mse_per_epoch = {x: 0 for x in _modes}
        epoch_loss = 0
        start_time = time.time()

        data_loader = {phase: torch.utils.data.DataLoader(datasets[phase], batch_size=batch_size[phase],
                                                          shuffle=True, pin_memory=True) for phase in datasets.keys()}
        lr = optimizer.param_groups[0]['lr']
        phase = 'train'
        for n_count, batch_yx in enumerate(data_loader[phase]):
            optimizer.zero_grad()
            if cuda:
                im_noisy, im_gt = batch_yx[0].cuda(), batch_yx[1].cuda()
            else:
                im_noisy, im_gt = batch_yx[0], batch_yx[1]
            phi_Z = model(im_noisy)
            # out_denoise = im_noisy - phi_Z
            out_denoise = phi_Z
            loss = criterion(out_denoise, im_gt)
            # loss = criterion(phi_Z, im_noisy-im_gt)
            epoch_loss += loss.item()

            # clip the gradnorm
            total_norm_D = nn.utils.clip_grad_norm_(param_D, clip_grad_D)

            loss.backward()
            optimizer.step()
            loss_per_epoch['Loss'] += loss.item() / num_iter_epoch[phase]
            if n_count % args.display == 0:
                print('%4d %4d / %4d loss = %2.4f' % (
                    epoch + 1, n_count, 100 * args.batch_size // batch_size[phase], loss.item() / batch_size[phase]))

            im_denoise = out_denoise.detach().data
            mse = F.mse_loss(im_denoise, im_gt)
            im_denoise.clamp_(0.0, 1.0)
            mse_per_epoch[phase] += mse
            if (n_count + 1) % args.print_freq == 0:
                log_str = '[Epoch:{:>2d}/{:<2d}] {:s}:{:0>4d}/{:0>4d}, mse={:.2e}, lr={:.1e}'
                print(log_str.format(epoch + 1, args.epoch, phase, n_count + 1, num_iter_epoch[phase],
                                     mse,lr))
        mse_per_epoch[phase] /= (n_count + 1)
        log_str = '{:s}: Loss={:+.2e}, mse={:.3e}'
        print(log_str.format(phase, loss_per_epoch['Loss'], mse_per_epoch[phase]))
        print('-' * 150)

        # test stage
        model.eval()
        psnr_per_epoch = {x: 0 for x in _modes[1:]}
        ssim_per_epoch = {x: 0 for x in _modes[1:]}
        for phase in _modes[1:]:
            for ii, data in enumerate(data_loader[phase]):
                im_noisy, im_gt = [x.cuda() for x in data]
                with torch.set_grad_enabled(False):
                    out_denoise = model(im_noisy)
                im_denoise = torch.clamp(out_denoise, 0.0, 1.0)
                mse = F.mse_loss(im_denoise, im_gt)
                mse_per_epoch[phase] += mse
                psnr_iter = batch_PSNR(im_denoise, im_gt)
                ssim_iter = batch_SSIM(im_denoise, im_gt)
                psnr_per_epoch[phase] += psnr_iter
                ssim_per_epoch[phase] += ssim_iter
                # print statistics every log_interval mini_batches
                if (ii + 1) % 20 == 0:
                    log_str = '[Epoch:{:>3d}/{:<3d}] {:s}:{:0>5d}/{:0>5d}, mse={:.2e}, ' + \
                              'psnr={:4.2f}, ssim={:5.4f}'
                    print(log_str.format(epoch + 1, args.epoch, phase, ii + 1, num_iter_epoch[phase],
                                         mse, psnr_iter, ssim_iter))

            psnr_per_epoch[phase] /= (ii + 1)
            ssim_per_epoch[phase] /= (ii + 1)
            mse_per_epoch[phase] /= (ii + 1)
            log_str = '{:s}: mse={:.3e}, PSNR={:4.2f}, SSIM={:5.4f}'
            print(log_str.format(phase, mse_per_epoch[phase], psnr_per_epoch[phase],
                                 ssim_per_epoch[phase]))
            print('-' * 90)

        elapsed_time = time.time() - start_time
        log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch + 1, epoch_loss / n_count, elapsed_time))
        np.savetxt('train_result.txt', np.hstack((epoch + 1, epoch_loss / n_count, elapsed_time)), fmt='%2.4f')
        # torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch + 1)))
