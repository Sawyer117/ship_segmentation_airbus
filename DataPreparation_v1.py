# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 13:23:41 2018

@author: Wen
"""
# %%

BATCH_SIZE = 64
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
# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
# General Read-in of all the available data
from utils import rle_encode
#masks = pd.read_csv(os.path.join('./',
#                                 'train_ship_segmentations.csv'))
masks = pd.read_csv('./train_ship_segmentations.csv')
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
unique_img_ids = masks.groupby('ImageId').size().reset_index(name='counts')
train_ids, valid_ids = train_test_split(unique_img_ids, 
                 test_size = 0.2, 
                 stratify = unique_img_ids['counts'])
train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, valid_ids)
print(train_df.shape[0], 'training masks')
print(valid_df.shape[0], 'validation masks')

train_df['counts'] = train_df.apply(lambda c_row: c_row['counts'] if 
                                    isinstance(c_row['EncodedPixels'], str) else
                                    0, 1)
train_df['counts'].hist()
# %%
# Undersample Empty Images-----------------------------------------------------
#Here we undersample the empty images to get a better balanced group with more ships to try and segment
train_df.groupby('counts').size()
train_df['triple_boats'] = train_df['counts'].map(lambda x: (x+2)//3)
balanced_train_df = train_df.groupby('triple_boats').apply(lambda x: x.sample(1000))
balanced_train_df['counts'].hist()

# Decode all the RLEs into Images----------------------------------------------
#We make a generator to produce batches of images
import cv2
from utils import masks_as_image
image_resize_value = 256
mask_resize_value = 256
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
            #out_rgb += [imread(rgb_path)]
            out_mask += [masks_as_image(c_masks['EncodedPixels'].values, mask_resize)]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask=[], []
                
train_gen = make_image_gen(balanced_train_df, image_resize_value, mask_resize_value)
valid_gen = make_image_gen(valid_df, image_resize_value, mask_resize_value)
train_x, train_y = next(train_gen)
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
from keras.preprocessing.image import ImageDataGenerator
dg_args = dict(featurewise_center = False, 
                  samplewise_center = False,
                  rotation_range = 15, 
                  width_shift_range = 0.1, 
                  height_shift_range = 0.1, 
                  shear_range = 0.01,
                  zoom_range = [0.9, 1.25],  
                  brightness_range = [0.5, 1.5],
                  horizontal_flip = True, 
                  vertical_flip = True,
                  fill_mode = 'reflect',
                   data_format = 'channels_last')

image_gen = ImageDataGenerator(**dg_args)
dg_args.pop('brightness_range')
label_gen = ImageDataGenerator(**dg_args)
# %%
# Augment Data-----------------------------------------------------------------

def create_aug_gen(in_gen, seed = None):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(255*in_x, 
                             batch_size = in_x.shape[0], 
                             seed = seed, 
                             shuffle=True)
        g_y = label_gen.flow(in_y, 
                             batch_size = in_x.shape[0], 
                             seed = seed, 
                             shuffle=True)

        yield next(g_x)/255.0, next(g_y)
        
cur_gen = create_aug_gen(train_gen)
t_x, t_y = next(cur_gen)
print('x', t_x.shape, t_x.dtype, t_x.min(), t_x.max())
print('y', t_y.shape, t_y.dtype, t_y.min(), t_y.max())
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.imshow(montage_rgb(t_x), cmap='gray')
ax1.set_title('images')
ax2.imshow(montage(t_y[:, :, :, 0]), cmap='gray_r')
ax2.set_title('ships')

# %%
# Training Setting---For Keras Only
#-------------------------------------------------------------
# =============================================================================
# from ModelBuild import unet
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
# 
# weight_path="{}_weights.best.hdf5".format('seg_model')
# 
# checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1, 
#                              save_best_only=True, mode='max', save_weights_only = True)
# 
# reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, 
#                                    patience=3, 
#                                    verbose=1, mode='max', epsilon=0.0001, cooldown=2, min_lr=1e-6)
# early = EarlyStopping(monitor="val_dice_coef", 
#                       mode="max", 
#                       patience=15) # probably needs to be more patient, but kaggle time is limited
# callbacks_list = [checkpoint, early, reduceLROnPlat]
# 
# seg_model = unet(t_x, adamlr= 1e-4, adamdecay = 0)
# =============================================================================
# %% Keras Version of Model
# Training Start Point --------------------------------------------------------

# =============================================================================
# loss_history = [seg_model.fit_generator(create_aug_gen(make_image_gen(balanced_train_df)), 
#                              steps_per_epoch=
#                                         min(100,
#                                             balanced_train_df.shape[0]//t_x.shape[0]), 
#                              epochs=24, 
#                              validation_data=valid_gen,
#                                         validation_steps = 25,
#                              callbacks=callbacks_list,
#                             workers=2)]
# =============================================================================
# %% Tensorflow Version of Model
import tensorflow as tf
from TfModel import TfModel
illustrate_results = True

datashape = [256,256]
[training_op, loss, Dice, keep_prob, logits, x, y] = TfModel(datashape)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

batch_size = BATCH_SIZE
num_of_epoch = 50
num_of_batches = 50
#saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_of_epoch):
        for batch_idx in range(num_of_batches):
            x_batch_train, y_batch_train = next(create_aug_gen(make_image_gen(balanced_train_df, image_resize_value, mask_resize_value)))
            #x_batch_train, y_batch_train = next(data_gen_small(x_train, y_train, batch_size))
            train_op, train_loss, dice = sess.run([training_op, loss, Dice], feed_dict={x: x_batch_train, y: y_batch_train, keep_prob: 0.5})
            
             
            x_batch_val, y_batch_val = next(valid_gen)
            val_op, val_loss = sess.run([training_op, loss], feed_dict={x: x_batch_val, y: y_batch_val, keep_prob: 0.5})
        
            print("epoch:", epoch, "\tbatch", batch_idx, "\tTraining Loss: {:.5f}".format(train_loss), "\tTesting Loss: {:.5f}".format(val_loss), "\tDice: {:.5f}".format(dice))
            if batch_idx % 10 == 0 and illustrate_results:
                logit_img = sess.run(logits, feed_dict={x: x_batch_val[2:3, :, :, :], keep_prob: 1})
                
                fig, ax = plt.subplots(1, 3, figsize=(12, 6))
                ax[0].imshow(x_batch_val[2, :, :, 0], cmap='gray')
                ax[1].imshow(y_batch_val[2, :, :, 0], cmap='gray')
                ax[2].imshow(logit_img[0, :, :, 0], cmap='gray')
                plt.show()
    saver.save(sess, './sUnet-ShipDetection1')

# %%
# Show loss-------------------------------------------------------------------- 
def show_loss(loss_history):
    epich = np.cumsum(np.concatenate(
        [np.linspace(0.5, 1, len(mh.epoch)) for mh in loss_history]))
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 10))
    _ = ax1.plot(epich,
                 np.concatenate([mh.history['loss'] for mh in loss_history]),
                 'b-',
                 epich, np.concatenate(
            [mh.history['val_loss'] for mh in loss_history]), 'r-')
    ax1.legend(['Training', 'Validation'])
    ax1.set_title('Loss')

    _ = ax2.plot(epich, np.concatenate(
        [mh.history['true_positive_rate'] for mh in loss_history]), 'b-',
                     epich, np.concatenate(
            [mh.history['val_true_positive_rate'] for mh in loss_history]),
                     'r-')
    ax2.legend(['Training', 'Validation'])
    ax2.set_title('True Positive Rate\n(Positive Accuracy)')
    
    _ = ax3.plot(epich, np.concatenate(
        [mh.history['binary_accuracy'] for mh in loss_history]), 'b-',
                     epich, np.concatenate(
            [mh.history['val_binary_accuracy'] for mh in loss_history]),
                     'r-')
    ax3.legend(['Training', 'Validation'])
    ax3.set_title('Binary Accuracy (%)')
    
    _ = ax4.plot(epich, np.concatenate(
        [mh.history['dice_coef'] for mh in loss_history]), 'b-',
                     epich, np.concatenate(
            [mh.history['val_dice_coef'] for mh in loss_history]),
                     'r-')
    ax4.legend(['Training', 'Validation'])
    ax4.set_title('DICE')

#show_loss(loss_history)

# Model Save
#seg_model.load_weights(weight_path)
#seg_model.save('seg_model.h5')

# %%

test_paths = os.listdir(test_image_dir)
print(len(test_paths))

fig, m_axs = plt.subplots(8, 2, figsize = (10, 40))
for (ax1, ax2), c_img_name in zip(m_axs, test_paths):
    c_path = os.path.join(test_image_dir, c_img_name)
    c_img = cv2.resize(imread(c_path),(256,256),interpolation=cv2.INTER_CUBIC)
    #c_img = imread(c_path)
    first_img = np.expand_dims(c_img, 0)/255.0
    #first_img = c_img
    first_seg = seg_model.predict(first_img)
    ax1.imshow(first_img[0])
    ax1.set_title('Image')
    ax2.imshow(first_seg[0, :, :, 0]*255)
    ax2.set_title('Prediction')
fig.savefig('test_predictions.png')

# %%
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
 

test_paths = os.listdir(test_image_dir)
print(len(test_paths))
fig, m_axs = plt.subplots(1, 2, figsize = (10, 40))
Cprogress=0
RLEValue = []

test_files = [f for f in os.listdir(test_image_dir) ]
submission_df = pd.DataFrame({'ImageId':test_files})
submission_df['EncodedPixels']=None
submission_df.to_csv('submission.csv', index=False)
submission_df.sample(3)

for item in test_paths:
    rgb_path1 = os.path.join(test_image_dir, item)
    out_rgb1 = cv2.resize(imread(rgb_path1),(256,256),interpolation=cv2.INTER_CUBIC)
    first_img = np.expand_dims(c_img, 0)/255.0
    #first_img = c_img
    first_seg = seg_model.predict(first_img)
    Return_img = cv2.resize(first_seg[0, :, :, 0],(768,768),interpolation=cv2.INTER_CUBIC)
    RLEValue += [rle_encode(Return_img)]
    ax1.imshow(first_img[0])
    ax1.set_title('Image')
    ax2.imshow(first_seg[0, :, :, 0])
    ax2.set_title('Prediction')
    print('%.3f%%' % (Cprogress/len(test_paths) * 100))
    #print(Cprogress/len(test_paths))
    Cprogress += 1