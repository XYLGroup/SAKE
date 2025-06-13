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
from config.configs import Config
from model.model import SAKE
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

def train(conf, lr_image, hr_image):
    ''' trainer for SAKE, etc.'''
    model = SAKE(conf, lr_image, hr_image)
    kernel, sr = model.train()
    return kernel, sr
def create_params(filename, args):
    ''' pass parameters to Config '''
    params = ['--model', args.model,
              '--input_image_path', args.input_dir + '/' + filename,
              '--sf', args.sf]
    return params


def main():
    I_loop_x = 5
    I_loop_k = 3
    D_loop = 5
    dataset = 'Salinas'
    method = 'SAKE'

    prog = argparse.ArgumentParser()
    prog.add_argument('--model', type=str, default=method, help='models: SAKE.')
    prog.add_argument('--dataset', '-d', type=str, default=dataset,
                      help='dataset, e.g., WDC, Pavia, Salinas, Indian_pines, Houston')
    prog.add_argument('--sf', type=str, default='4', help='The wanted SR scale factor')

    args = prog.parse_args()

    args.input_dir = 'data/datasets/{}/SAKE_lr_x{}'.format(args.dataset, args.sf)
    filesource = os.listdir(os.path.abspath(args.input_dir))
    filesource.sort()
    now_time = str(datetime.datetime.now())[:-10].replace(':', '-')

    for filename in filesource[:]:
        print(filename)

        data = scipy.io.loadmat(os.path.join(args.input_dir, filename))
        HR = scipy.io.loadmat('data/datasets/{}/HR/{}'.format(args.dataset, filename))
        Ch = data['input'].shape[2]
        print('Channels:', Ch)

        image = data['input']
        hr_img = HR['gt']

        # setting the parameters
        conf = Config().parse(create_params(filename, args))
        conf, args = parameters_setting(conf, args, I_loop_x, I_loop_k, D_loop, method, filename)

        lr_image = np2tensor01(image).cuda().unsqueeze(0)


        hr_image = np2tensor01(hr_img).unsqueeze(0)

        kernel, sr_dip = train(conf, lr_image, hr_image)
        sr_im = tensor2np(sr_dip)
        savemat(os.path.join(conf.output_dir_path, "{}_SR.mat".format(args.dataset)), {'img': sr_im})

        image_psnr, im_ssim, kernel_psnr = evaluation_dataset(Ch, args.input_dir, conf.output_dir_path, args.dataset, conf)


if __name__ == '__main__':
    seed = 2
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    main()
