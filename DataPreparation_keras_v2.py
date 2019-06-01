# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 13:23:41 2018

@author: Wen
"""
# %%

BATCH_SIZE = 24
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.util import montage
montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
ship_dir = './input'
train_image_dir = os.path.join(ship_dir, 'train_v2')
test_image_dir = os.path.join(ship_dir, 'test_v2')

exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg', 
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg'] #corrupted images

   
# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
# General Read-in of all the available data
#from utils import rle_encode
#masks = pd.read_csv(os.path.join('./',
#                                 'train_ship_segmentations.csv'))
masks = pd.read_csv('./train_ship_segmentations_v2.csv')
print(masks.shape[0], 'masks found')
print(masks['ImageId'].value_counts().shape[0])
masks.head()

#os.environ['CUDA_VISIBLE_DEVICES'] = ''  ## enable this to use tensorflow CPU version instead of GPU

# delete empty ships here
masks = masks[masks.EncodedPixels.notnull()]
# %%
# Split into training and validation groups------------------------------------
# We stratify by the number of boats appearing so we have nice balances in each set

from sklearn.model_selection import train_test_split
#unique_img_ids = masks.groupby('ImageId').size().reset_index(name='counts')
#train_ids, valid_ids = train_test_split(unique_img_ids, 
#                 test_size = 0.2, 
#                 stratify = unique_img_ids['counts'])
#train_df = pd.merge(masks, train_ids)
#valid_df = pd.merge(masks, valid_ids)
#print(train_df.shape[0], 'training masks')
#print(valid_df.shape[0], 'validation masks')

#train_df['counts'] = train_df.apply(lambda c_row: c_row['counts'] if 
#                                    isinstance(c_row['EncodedPixels'], str) else
#                                    0, 1)
#train_df['counts'].hist()


masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
for el in exclude_list:
    if(el in unique_img_ids['ImageId']): 
        print('corruption found:', el) 
        unique_img_ids.drop(['el'])

unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])
masks.drop(['ships'], axis=1, inplace=True)
train_ids, valid_ids = train_test_split(unique_img_ids, 
                 test_size = 0.2, 
                 stratify = unique_img_ids['ships'])
train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, valid_ids)
print(train_df.shape[0], 'training masks')
print(valid_df.shape[0], 'validation masks')
train_df[['ships', 'has_ship']].hist()
# %%
# Undersample Empty Images-----------------------------------------------------
#Here we undersample the empty images to get a better balanced group with more ships to try and segment
#train_df.groupby('counts').size()
#train_df['triple_boats'] = train_df['counts'].map(lambda x: (x+2)//3)
#balanced_train_df = train_df.groupby('triple_boats').apply(lambda x: x.sample(1000))
#balanced_train_df['counts'].hist()
balanced_train_df = train_df
# Decode all the RLEs into Images----------------------------------------------
#We make a generator to produce batches of images
import cv2
from utils import masks_as_image
image_resize_value = 192
mask_resize_value = 192
def make_image_gen(in_df, image_resize, mask_resize):
    all_batches = list(in_df.groupby('ImageId'))
    batch_size = BATCH_SIZE
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(train_image_dir, c_img_id)
            out_rgb += [cv2.resize(imread(rgb_path),(image_resize,image_resize),interpolation=cv2.INTER_CUBIC)]
            out_mask += [masks_as_image(c_masks['EncodedPixels'].values, mask_resize)]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask=[], []
                
train_gen = make_image_gen(balanced_train_df, image_resize_value, mask_resize_value)
valid_gen = make_image_gen(valid_df, image_resize_value, mask_resize_value)
train_x, train_y = next(train_gen)
valid_x, valid_y = next(valid_gen)
print('x', train_x.shape, train_x.min(), train_x.max())
print('y', train_y.shape, train_y.min(), train_y.max())
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (30, 10))
batch_rgb = montage_rgb(train_x)
batch_seg = montage(train_y[:, :, :, 0])
ax1.imshow(batch_rgb)
ax1.set_title('Images')
ax2.imshow(batch_seg)
ax2.set_title('Segmentations')
ax3.imshow(mark_boundaries(batch_rgb, 
                           batch_seg.astype(int)))
ax3.set_title('Outlined Ships')
fig.savefig('overview.png')

# %%
# Augment Data-----------------------------------------------------------------
from keras.preprocessing.image import ImageDataGenerator
dg_args = dict(featurewise_center = False, 
                  samplewise_center = False,
                  rotation_range = 15, 
                  width_shift_range = 0.1, 
                  height_shift_range = 0.1, 
                  shear_range = 0.01,
                  zoom_range = [0.9, 1.25],
                  horizontal_flip = True, 
                  vertical_flip = True,
                  fill_mode = 'reflect',
                   data_format = 'channels_last')

image_gen = ImageDataGenerator(**dg_args)
label_gen = ImageDataGenerator(**dg_args)

def create_aug_gen(in_gen, seed = None):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(255*in_x, batch_size=in_x.shape[0], seed = seed, shuffle = True)
        g_y = image_gen.flow(in_y, batch_size=in_x.shape[0], seed = seed, shuffle = True)
        
        yield next(g_x)/255.0, next(g_y)
# =============================================================================

# =============================================================================
        
cur_gen = create_aug_gen(train_gen)  #use augmentation
#cur_gen = train_gen  #unabling the augmentation
t_x, t_y = next(cur_gen)
print('x', t_x.shape, t_x.dtype, t_x.min(), t_x.max())
print('y', t_y.shape, t_y.dtype, t_y.min(), t_y.max())
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.imshow(montage_rgb(t_x), cmap='gray')
ax1.set_title('images')
ax2.imshow(montage(t_y[:, :, :, 0]), cmap='gray_r')
ax2.set_title('ships')


# =============================================================================
# %%    Version of Model
from segmentation_models import Unet
#from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import dice_loss
from keras.losses import binary_crossentropy
from segmentation_models.metrics import iou_score
from segmentation_models.metrics import f_score
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
import tensorflow as tf
from Model_Keras import unet

def dice_coef(y_true, y_pred, smooth=1e-8): #smooth was 1
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2.0 * intersection)/(union + smooth), axis=0)
    #return K.mean((2.0 * intersection+smooth)/(union + smooth), axis=0)

def dice_p_bce(in_gt, in_pred):
    return 1e-3 * binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)

def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true) * K.flatten(K.round(y_pred))) / K.sum(y_true)

def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))

def weighted_bce_loss(y_true, y_pred, weight):
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    loss = weight * (logit_y_pred * (1. - y_true) + 
                     K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd
    averaged_mask = K.pool2d(
            y_true, pool_size=(50, 50), strides=(1, 1), padding='same', pool_mode='avg')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight = 5. * K.exp(-5. * K.abs(averaged_mask - 0.5))
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + dice_loss(y_true, y_pred)
    return loss

def focal_loss(y_true, y_pred,gamma=2., alpha=.25):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def IoU(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return -K.mean( (intersection + eps) / (union + eps), axis=0)
# %%
weight_path="{}_weights.best.hdf5".format('seg_model')
checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max', save_weights_only=False)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.33,
                                   patience=1, verbose=1, mode='min',
                                   min_delta=0.0001, cooldown=0, min_lr=1e-8)
early = EarlyStopping(monitor="val_loss", mode="min", verbose=2,
                      patience=20) # probably needs to be more patient, but kaggle time is limit
callbacks_list = [checkpoint, reduceLROnPlat,early]


# define model
#BACKBONE = 'resnet34'
#seg_model = Unet(BACKBONE, activation='relu',encoder_weights=None,encoder_freeze=False)
# Above model from https://github.com/qubvel/segmentation_models

# Custom model- Unet
seg_model = unet(t_x)

#seg_model.compile(optimizer=Adam(1e-3, decay=0), loss=bce_jaccard_loss,dice_loss, metrics=[iou_score])
seg_model.compile(optimizer=Adam(1e-3, decay=1e-6 ), loss=dice_loss, metrics=[f_score,dice_coef,true_positive_rate])

# fit model
aug_gen = create_aug_gen(make_image_gen(balanced_train_df, image_resize_value, mask_resize_value))

seg_model.load_weights(weight_path)  #use this line to load from auto-saved weights during training
#seg_model.load_weights('seg_model_Unet_vgg16_v2.h5')
seg_model.save('seg_model_Unet34_IoU.h5') 

loss_history = [seg_model.fit_generator(aug_gen,
             steps_per_epoch=300,                    #train_df.shape[0]//BATCH_SIZE
             epochs=50,
             validation_data=(valid_x, valid_y),
             callbacks=callbacks_list,verbose=1,
            # the generator is not very thread safe
                       )]  



# %%
# Show loss-------------------------------------------------------------------- 
def show_loss(loss_history):
    epochs = np.concatenate([mh.epoch for mh in loss_history])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    
    _ = ax1.plot(epochs, np.concatenate([mh.history['loss'] for mh in loss_history]), 'b-',
                 epochs, np.concatenate([mh.history['val_loss'] for mh in loss_history]), 'r-')
    ax1.legend(['Training', 'Validation'])
    ax1.set_title('Loss')
    
    _ = ax2.plot(epochs, np.concatenate([mh.history['f_score'] for mh in loss_history]), 'b-',
                 epochs, np.concatenate([mh.history['val_f_score'] for mh in loss_history]), 'r-')
    ax2.legend(['Training', 'Validation'])
    ax2.set_title('f_score (%)')

show_loss(loss_history)

# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (30, 10))
batch_rgb_valid = montage_rgb(valid_x)
batch_seg_valid = montage(valid_y[:, :, :, 0])
ax1.imshow(mark_boundaries(batch_rgb_valid, 
                           batch_seg_valid.astype(int)))
ax1.set_title('Original Images(with Mask boundaries)')
ax2.imshow(batch_seg_valid)
ax2.set_title('Segmentations(Ground Truth)')
output = seg_model.predict(valid_x[:,:,:,:])
batch_output = montage(output[:, :, :, 0])
ax3.imshow(batch_output)
ax3.set_title('Prediction')
fig.savefig('overview.png')
    
# %%

test_paths = os.listdir(test_image_dir)
print(len(test_paths))

fig, m_axs = plt.subplots(8, 2, figsize = (10, 40))
for (ax1, ax2), c_img_name in zip(m_axs, test_paths):
    c_path = os.path.join(test_image_dir, c_img_name)
    c_img = cv2.resize(imread(c_path),(image_resize_value,image_resize_value),interpolation=cv2.INTER_CUBIC)
    first_img = np.expand_dims(c_img, 0)/255.0
    #first_img = c_img
    first_seg = seg_model.predict(first_img)
    ax1.imshow(first_img[0])
    ax1.set_title('Image')
    ax2.imshow(first_seg[0, :, :, 0]*255)
    ax2.set_title('Prediction')
fig.savefig('test_predictions.png')

# =============================================================================
# # %%
# import pandas as pd
# from pandas import ExcelWriter
# from pandas import ExcelFile
# import numpy as np
#  
# 
# test_paths = os.listdir(test_image_dir)
# print(len(test_paths))
# fig, m_axs = plt.subplots(1, 2, figsize = (10, 40))
# Cprogress=0
# RLEValue = []
# 
# test_files = [f for f in os.listdir(test_image_dir) ]
# submission_df = pd.DataFrame({'ImageId':test_files})
# submission_df['EncodedPixels']=None
# submission_df.to_csv('submission.csv', index=False)
# submission_df.sample(3)
# 
# for item in test_paths:
#     rgb_path1 = os.path.join(test_image_dir, item)
#     out_rgb1 = cv2.resize(imread(rgb_path1),(256,256),interpolation=cv2.INTER_CUBIC)
#     first_img = np.expand_dims(c_img, 0)/255.0
#     #first_img = c_img
#     first_seg = seg_model.predict(first_img)
#     Return_img = cv2.resize(first_seg[0, :, :, 0],(768,768),interpolation=cv2.INTER_CUBIC)
#     RLEValue += [rle_encode(Return_img)]
#     ax1.imshow(first_img[0])
#     ax1.set_title('Image')
#     ax2.imshow(first_seg[0, :, :, 0])
#     ax2.set_title('Prediction')
#     print('%.3f%%' % (Cprogress/len(test_paths) * 100))
#     #print(Cprogress/len(test_paths))
#     Cprogress += 1
# =============================================================================
