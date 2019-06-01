# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:25:18 2018

@author: Wen
"""
EDGE_CROP = 16
GAUSSIAN_NOISE = 0.1
filtersize = 16

from keras import models, layers
import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
def dice_p_bce(in_gt, in_pred):
    return 0.01*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)
def dice_loss(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
    return 1-dice
def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)


# Build U-Net model
def unet(t_x, adamlr,adamdecay):
    input_img = layers.Input(t_x.shape[1:], name = 'RGB_Input')
    pp_in_layer = layers.GaussianNoise(GAUSSIAN_NOISE)(input_img)
    pp_in_layer = layers.BatchNormalization()(pp_in_layer)
    
    c1 = layers.Conv2D(filtersize, (3, 3), activation='relu', padding='same') (pp_in_layer)
    c1 = layers.Conv2D(filtersize, (3, 3), activation='relu', padding='same') (c1)
    p1 = layers.MaxPooling2D((2, 2)) (c1)
    
    c2 = layers.Conv2D(filtersize*2, (3, 3), activation='relu', padding='same') (p1)
    c2 = layers.Conv2D(filtersize*2, (3, 3), activation='relu', padding='same') (c2)
    p2 = layers.MaxPooling2D((2, 2)) (c2)
    
    c3 = layers.Conv2D(filtersize*4, (3, 3), activation='relu', padding='same') (p2)
    c3 = layers.Conv2D(filtersize*4, (3, 3), activation='relu', padding='same') (c3)
    p3 = layers.MaxPooling2D((2, 2)) (c3)
    
    c4 = layers.Conv2D(filtersize*8, (3, 3), activation='relu', padding='same') (p3)
    c4 = layers.Conv2D(filtersize*8, (3, 3), activation='relu', padding='same') (c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2)) (c4)
    
    
    c5 = layers.Conv2D(filtersize*16, (3, 3), activation='relu', padding='same') (p4)
    c5 = layers.Conv2D(filtersize*16, (3, 3), activation='relu', padding='same') (c5)
    
    u6 = layers.Conv2DTranspose(filtersize*8, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(filtersize*8, (3, 3), activation='relu', padding='same') (u6)
    c6 = layers.Conv2D(filtersize*8, (3, 3), activation='relu', padding='same') (c6)
    
    u7 = layers.Conv2DTranspose(filtersize*4, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(filtersize*4, (3, 3), activation='relu', padding='same') (u7)
    c7 = layers.Conv2D(filtersize*4, (3, 3), activation='relu', padding='same') (c7)
    
    u8 = layers.Conv2DTranspose(filtersize*2, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(filtersize*2, (3, 3), activation='relu', padding='same') (u8)
    c8 = layers.Conv2D(filtersize*2, (3, 3), activation='relu', padding='same') (c8)
    
    u9 = layers.Conv2DTranspose(filtersize, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(filtersize, (3, 3), activation='relu', padding='same') (u9)
    c9 = layers.Conv2D(filtersize, (3, 3), activation='relu', padding='same') (c9)
    #c9 = layers.Conv2D(2, (3, 3), activation='relu', padding='same') (c9)
    
    d = layers.Conv2D(1, (1, 1), activation='sigmoid') (c9)
    #d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
    #d = layers.ZeroPadding2D((EDGE_CROP, EDGE_CROP))(d)
    
    seg_model = models.Model(inputs=[input_img], outputs=[d])
    seg_model.summary()
    seg_model.compile(optimizer=Adam(adamlr, decay=adamdecay), loss=dice_loss, metrics=[dice_coef, 'binary_accuracy', true_positive_rate])


    return seg_model