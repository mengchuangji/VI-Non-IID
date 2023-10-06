#!/usr/bin/env python
# -*- coding:utf-8 -*-


import argparse

def set_opts():
    parser = argparse.ArgumentParser()
    # trainning settings
    parser.add_argument('--batch_size', type=int, default=64,
                                                         help="Batchsize of training, (default:64)")
    # parser.add_argument('--patch_size', type=int, default=128,
    #                                                help="Patch size of data sample,  (default:128)")
    parser.add_argument('--epochs', type=int, default=50, help="Training epohcs,  (default:60)")
    parser.add_argument('--lr', type=float, default=2e-4,
                                                  help="Initialized learning rate, (default: 2e-4)")
    parser.add_argument('--gamma', type=float, default=0.5,
                                         help="Decaying rate for the learning rate, (default: 0.5)")
    parser.add_argument('-p', '--print_freq', type=int, default=500,
                                                              help="Print frequence (default: 100)")
    parser.add_argument('-s', '--save_model_freq', type=int, default=1,
                                                            help="Save moel frequence (default: 1)")

    # Cliping the Gradients Norm during the training
    parser.add_argument('--clip_grad_D', type=float, default=1e4,
                                             help="Cliping the gradients for D-Net, (default: 1e4)")
    parser.add_argument('--clip_grad_S', type=float, default=1e3,
                                             help="Cliping the gradients for S-Net, (default: 1e3)")

    # GPU settings
    parser.add_argument('--gpu_id', type=int, nargs='+', default=0,
                                                           help="GPU ID, which allow multiple GPUs")
    # [0, 1, 2, 3, 4, 5, 6, 7]
    # dataset settings
    parser.add_argument('--SIDD_dir', default='/home/mcj/mcj/datasets/SIDD_mcj/', type=str, metavar='PATH',
                                              help="Path to save the SIDD dataset, (default: None)")
    #mcj
    parser.add_argument('--simulate_dir', default='/home/mcj/mcj/datasets/Waterloo', type=str,
                        metavar='PATH', help="Path to save the images, (default: None)")

    # model and log saving
    parser.add_argument('--log_dir', default='./log', type=str, metavar='PATH',
                                                 help="Path to save the log file, (default: ./log)")
    parser.add_argument('--model_dir', default='./model', type=str, metavar='PATH',
                                             help="Path to save the model file, (default: ./model)")
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                                               help="Path to the latest checkpoint (default: None)")
    # parser.add_argument('--resume', default='./model/model_29', type=str, metavar='PATH',
    #                                            help="Path to the latest checkpoint (default: None)")
    parser.add_argument('--num_workers', default=0, type=int,
                                                help="Number of workers to load data, (default: 8)")
    # hyper-parameters --eps2 default=1e-6
    parser.add_argument('--eps2', default=1e-6, type=float,
                                                    help="Variance for prior of Z, (default: 1e-6)")
    parser.add_argument('--alpha', default=1, type=float,
                        help="alpha for KL_vi, (default: 1)")
    parser.add_argument('--beta', default=1, type=float,
                        help="beta for KL_sigma, (default: 1)")
    parser.add_argument('--radius', default=3, type=int,
                                                help="Radius for the Gaussian filter, (default: 3)")

    # network architecture
    parser.add_argument('--net', type=str, default='VDN',
                               help="Network architecture: VDN, VDNRD or VDNRDU, (default:VDN)")
    parser.add_argument('--slope', type=float, default=0.2,
                                                 help="Initial value for LeakyReLU, (default: 0.2)")
    parser.add_argument('--wf', type=int, default=64,
                                                   help="Initilized filters of UNet, (default: 64)")
    parser.add_argument('--depth', type=int, default=4, help="The depth of UNet, (default: 4)")


    ##########
    # parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
    parser.add_argument('--data_dir', default='/home/shendi_mcj/datasets/seismic/train', type=str, help='path of train data')
    parser.add_argument('--sigma', default=25, type=int, help='noise level')
    # parser.add_argument('--epoch', default=50, type=int, help='number of train epoches')
    # parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
    # parser.add_argument('--batch_size', default=50, type=int, help='batch size')
    parser.add_argument('--patch_size', default=(32, 32), type=int, help='patch size')#dncnn
    parser.add_argument('--stride', default=(32, 32), type=int, help='the step size to slide on the data')
    parser.add_argument('--jump', default=3, type=int, help='the space between shot')
    parser.add_argument('--download', default=False, type=bool,
                        help='if you will download the dataset from the internet')
    parser.add_argument('--datasets', default=0, type=int,
                        help='the num of datasets you want be download,if download = True')
    parser.add_argument('--train_data_num', default=100000, type=int, help='the num of the train_data')
    parser.add_argument('--aug_times', default=0, type=int, help='Number of aug operations')
    parser.add_argument('--scales', default=[1], type=list, help='data scaling')
    parser.add_argument('--agc', default=True, type=int, help='Normalize each trace by amplitude')
    parser.add_argument('--verbose', default=True, type=int, help='Whether to output the progress of data generation')
    parser.add_argument('--display', default=1000, type=int, help='interval for displaying loss')

    args = parser.parse_args()

    return args
def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')