# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 12:58:02 2018

@author: Wen
"""
import numpy as np
import cv2
import os
from skimage.io import imread

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list, mask_size):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((mask_size, mask_size), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += cv2.resize(rle_decode(mask),(mask_size,mask_size),interpolation=cv2.INTER_CUBIC)
            #all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)

def return_image_BATCH(image_dir, image_resize, mask_resize,BATCH_SIZE):
    pin = True
    batch_size = BATCH_SIZE
    out_rgb = []
    NofF = len([name for name in os.listdir(image_dir)])
    while pin:
        for file in os.listdir(image_dir):
            rgb_path = os.path.join(image_dir, file)
            out_rgb += [cv2.resize(imread(rgb_path),(image_resize,image_resize),interpolation=cv2.INTER_CUBIC)]
            if len(out_rgb)>=batch_size:
                out_rgb = np.stack(out_rgb, 0)/255.0
                pin = False
    return out_rgb,NofF