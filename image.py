import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

def display_pseudo_color(image1):
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
    # plt.savefig('zao100')

    plt.show()

data = sio.loadmat('SAKE/data/datasets/foster2/foster2_crop.mat')['input']
display_pseudo_color(data)