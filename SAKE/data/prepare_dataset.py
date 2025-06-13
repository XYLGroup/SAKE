# Generate random Gaussian kernels and downscale images
import sys
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from scipy.ndimage import filters, measurements, interpolation, center_of_mass, shift
import glob
from scipy.io import savemat
import os
from PIL import Image
import torch
import torch.nn.functional as F
import argparse
from guided_diffusion.core import imresize
from math import log, sqrt


# Function for centering a kernel
def kernel_shift(kernel, k_size):
    # First calculate the current center of mass for the kernel
    current_center_of_mass = center_of_mass(kernel)

    # The idea kernel center
    # for image blurred by filters.correlate
    # wanted_center_of_mass = (np.array(kernel.shape) - sf) / 2.

    wanted_center_of_mass = k_size // 2

    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass

    # Finally shift the kernel and return
    return shift(kernel, shift_vec)


# Function for calculating the X4 kernel from the X2 kernel, used in KernelGAN
def analytic_kernel(k):
    k_size = k.shape[0]
    # Calculate the big kernels size
    big_k = np.zeros((3 * k_size - 2, 3 * k_size - 2))
    # Loop over the small kernel to fill the big one
    for r in range(k_size):
        for c in range(k_size):
            big_k[2 * r:2 * r + k_size, 2 * c:2 * c + k_size] += k[r, c] * k
    # Crop the edges of the big kernel to ignore very small values and increase run time of SR
    crop = k_size // 2
    cropped_big_k = big_k[crop:-crop, crop:-crop]
    # Normalize to 1
    return cropped_big_k / cropped_big_k.sum()


# Function for generating one fixed kernel
def gen_kernel_fixed(k_size, scale_factor, lambda_1, lambda_2, theta, noise):
    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1, lambda_2]);
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position (shifting kernel for aligned image)
    MU = k_size / 2.
    MU = MU[None, None, :, None]

    # Create meshgrid for Gaussian
    [X, Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z - MU
    ZZ_t = ZZ.transpose(0, 1, 3, 2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)

    # shift the kernel so it will be centered
    raw_kernel_centered = kernel_shift(raw_kernel, k_size)

    # Normalize the kernel and return
    kernel = raw_kernel_centered / np.sum(raw_kernel_centered)

    return kernel


# Function for generating one random kernel
def gen_kernel_random(k_size, scale_factor, min_var, max_var, noise_level):
    lambda_1 = min_var + np.random.rand() * (max_var - min_var)
    lambda_2 = min_var + np.random.rand() * (max_var - min_var)
    theta = np.random.rand() * np.pi
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2

    kernel = gen_kernel_fixed(k_size, scale_factor, lambda_1, lambda_2, theta, noise)

    return kernel


# Function for degrading one image
def degradation(input, kernel, scale_factor, noise_im, device=torch.device('cuda')):
    # preprocess image and kernel
    kernel_size = kernel.shape[0]
    Ch = input.shape[2]
    input = torch.from_numpy(input).to(device).permute(2, 0, 1).unsqueeze(0)
    kernel = torch.from_numpy(kernel).repeat(Ch, 1, 1, 1).to(device)

    # blur
    output = F.conv2d(input, weight=kernel, padding=int((kernel_size - 1) / 2), groups=Ch)

    # down-sample
    # output = output[:, :, 0::scale_factor[0], 0::scale_factor[0]]
    output = imresize(output, 1 / scale_factor[0]) # torch.Size([1, 191, 16, 16])
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # add AWGN noise
    output += np.random.normal(0, np.random.uniform(0, noise_im), output.shape)

    return output


def generate_dataset(images_path, out_path_im, out_path_ker, k_size, scale_factor, min_var, max_var, noise_ker,
                     noise_im, kernelgan_x4=False):
    os.makedirs(out_path_im, exist_ok=True)
    os.makedirs(out_path_ker, exist_ok=True)

    # Load images, downscale using kernels and save
    print(images_path)
    data = scipy.io.loadmat(images_path)['gt']

    if kernelgan_x4:
        # As in original kernelgan, for x4, we use analytic kernel calculated from DDS2M_result.
        kernel = gen_kernel_random(k_size, 2, min_var, max_var, noise_ker)
        kernel = analytic_kernel(kernel)
        kernel = kernel_shift(kernel, 4)
    else:
        kernel = gen_kernel_random(k_size, scale_factor, min_var, max_var, noise_ker)

    lr = degradation(data, kernel, scale_factor, noise_im,
                     device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    # save kernel
    savemat('%s/%s.mat' % (out_path_ker, os.path.splitext(os.path.basename(images_path))[0]), {'Kernel': kernel})
    plot_kernel(kernel, '%s/%s.png' % (out_path_ker, os.path.splitext(os.path.basename(images_path))[0]))

    # save lr_image
    savemat('%s/%s.mat' % (out_path_im, os.path.splitext(os.path.basename(images_path))[0]), {'input': lr})
    display_pseudo_color(lr)


def plot_kernel(gt_k_np, savepath):
    plt.clf()
    f, ax = plt.subplots(1, 1, figsize=(6, 4), squeeze=False)
    im = ax[0, 0].imshow(gt_k_np, vmin=0, vmax=gt_k_np.max())
    plt.colorbar(im, ax=ax[0, 0])
    # im = ax[0, 1].imshow(out_k_np, vmin=0, vmax=out_k_np.max())
    # plt.colorbar(im, ax=ax[0, 1])
    # ax[0, 0].set_title('GT')
    # ax[0, 1].set_title('PSNR: {:.2f}'.format(calculate_psnr(gt_k_np, out_k_np, True)))

    plt.savefig(savepath)

def display_pseudo_color(image1):
    plt.figure(figsize=(12, 6))

    def process_image(image):
        red_channel = image[:, :, 60]
        green_channel = image[:, :, 27]
        blue_channel = image[:, :, 17]

        pseudo_color_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)
        return pseudo_color_image

    pseudo_color_image1 = process_image(image1)
    plt.imshow(pseudo_color_image1)
    plt.axis('off')

    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SAKE',
                        help='SAKE, generate data blurred by anisotropic Gaussian kernel. '
                             'Note that kernelgan uses x4 analytical kernel calculated from DDS2M_result.')
    parser.add_argument('--sf', type=int, default=4, help='scale factor: 2, 3, 4, 8')
    parser.add_argument('--dataset', type=str, default='Salinas', help='dataset: Pavia, Salinas, WDC, Indian_pines, Houston')
    parser.add_argument('--noise_ker', type=float, default=0, help='noise on kernel, e.g. 0.4')
    parser.add_argument('--noise_im', type=float, default=0/255, help='noise on LR image, e.g. 10/255=0.039')
    opt = parser.parse_args()

    images_path = 'datasets/{}/HR/{}_crop.mat'.format(opt.dataset, opt.dataset)
    out_path_im = 'datasets/{}/{}_lr_x{}'.format(opt.dataset, opt.model, opt.sf)
    out_path_ker = 'datasets/{}/{}_gt_k_x{}'.format(opt.dataset, opt.model, opt.sf)

    if opt.model == 'SAKE':
        min_var = 0.4
        max_var = 5

        k_size = np.array([19, 19])

        generate_dataset(images_path, out_path_im, out_path_ker, k_size, np.array([opt.sf, opt.sf]), min_var, max_var,
                         opt.noise_ker, opt.noise_im)

    else:
        raise NotImplementedError

if __name__ == '__main__':
    seed = 4
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    main()
    sys.exit()

