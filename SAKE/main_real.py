import os
import argparse
import torch
import sys
import numpy as np
import warnings

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
from model.util import (map2tensor, tensor2np, evaluation_dataset, np2tensor01)
from config.configs_real import Config
from model.model_real import SAKE
import time
import datetime
from Settings import parameters_setting
import scipy.io
from scipy.io import savemat
import xlwt
import openpyxl
# for nonblind SR
sys.path.append('../')
from NonblindSR.usrnet import USRNet

'''
# ------------------------------------------------
# main.py for DIP-KP
# ------------------------------------------------
'''

def train(conf, lr_image):
    ''' trainer for SAKE, etc.'''
    model = SAKE(conf, lr_image)
    kernel, sr = model.train()
    return kernel, sr


def display_pseudo_color_01(image1):
    plt.figure(figsize=(12, 6))

    def process_image(image):
        red_channel = image[:, :, 27]
        green_channel = image[:, :, 11]
        blue_channel = image[:, :, 5]

        pseudo_color_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)
        return pseudo_color_image

    pseudo_color_image1 = process_image(image1)

    plt.imshow(pseudo_color_image1)
    plt.axis('off')
    plt.show()


def create_params(filename, args):
    ''' pass parameters to Config '''
    params = ['--model', args.model,
              '--input_image_path', args.input_dir + '/' + filename,
              '--sf', args.sf]
    if args.SR:
        params.append('--SR')
    return params


def main():
    I_loop_x = 5
    I_loop_k = 3
    D_loop = 5
    dataset = 'foster2'
    method = 'SAKE'

    prog = argparse.ArgumentParser()
    prog.add_argument('--model', type=str, default=method, help='models: SAKE.')
    prog.add_argument('--dataset', '-d', type=str, default=dataset,
                      help='dataset, e.g., WDC, Pavia, Salinas, Indian_pines, Houston')
    prog.add_argument('--sf', type=str, default='4', help='The wanted SR scale factor')
    prog.add_argument('--path-nonblind', type=str, default='data/pretCAVEy.matrained_models/usrnet_tiny.pth',
                      help='path for trained nonblind model')
    prog.add_argument('--SR', action='store_true', default=False, help='when activated - nonblind SR is performed')


    args = prog.parse_args()

    args.input_dir = 'data/datasets/{}'.format(args.dataset)
    filesource = os.listdir(os.path.abspath(args.input_dir))
    filesource.sort()
    now_time = str(datetime.datetime.now())[:-10].replace(':', '-')

    for filename in filesource[:]:
        print(filename)


        data = scipy.io.loadmat(os.path.join(args.input_dir, filename))['input']
        Ch = data.shape[2]
        print('Channels:', Ch)
        print(data.shape)

        # setting the parameters
        conf = Config().parse(create_params(filename, args))
        conf, args = parameters_setting(conf, args, I_loop_x, I_loop_k, D_loop, method, filename)

        lr_image = np2tensor01(data).cuda().unsqueeze(0)

        kernel, sr_dip = train(conf, lr_image)
        sr_im = tensor2np(sr_dip)
        savemat(os.path.join(conf.output_dir_path, "After_SR.mat"), {'img': sr_im})

        display_pseudo_color_01(sr_im)



if __name__ == '__main__':
    seed = 2
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    main()
