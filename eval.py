import os
import cv2
import torch
from piq import ssim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def getSSIM(im_dir):
    images = sorted(os.listdir(im_dir), key = lambda x: int(x[:-4]))
    SSIM = []
    for f1, f2 in zip(images, images[1:]):
        ref_im = cv2.imread(os.path.join(im_dir, f1))
        ref_im = torch.from_numpy(ref_im).permute(2, 0, 1)[None, ...]
        float_im = cv2.imread(os.path.join(im_dir, f2))
        float_im = torch.from_numpy(float_im).permute(2, 0, 1)[None, ...]
        
        metric = ssim(float_im, ref_im, data_range = 255)            # Creat SSIM metric
        SSIM.append(metric.item())
    return SSIM


if __name__ == "__main__":
    # image path
    im_dir_orig = 'images_janan'
    im_dir_adam = 'images_painted'
    im_dir_scaled = 'scaled_images'
    num_images = len(os.listdir(im_dir_scaled))
    x = np.arange(num_images)

    # SSIM_orig = getSSIM(im_dir_orig)
    # SSIM_adam = getSSIM(im_dir_adam)
    SSIM_scaled = getSSIM(im_dir_scaled)

    # dataframe = pd.DataFrame({'SSIM_original':SSIM_orig, 'SSIM_adam':SSIM_adam})
    dataframe = pd.DataFrame({'SSIM_scaled':SSIM_scaled})
    dataframe.to_csv("SSIM.csv", mode='a', index=False, sep=',')




    

