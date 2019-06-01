# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 13:12:12 2018

@author: Wen
"""
import tensorflow as tf
import tensorlayer as tl

def TfModel(t_x):
    height, width = t_x[0], t_x[1]
    msk_height = 256
    
    x = tf.placeholder(tf.float32, [None, height, width, 3], name='x')
    y = tf.placeholder(tf.float32, [None, msk_height, msk_height, 1], name='y')
    
    keep_prob = tf.placeholder(tf.float32, name='drop_out')
    
    #input_layer = tf.keras.layers.Input(shape=[height, width])
    c1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(x)
    c12 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(c1)
    mc12 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(c12)

    # second layer
    c2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(mc12)
    c22 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(c2)
    mc22 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(c22)
    
    # third layer
    c3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(mc22)
    c32 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(c3)
    mc32 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(c32)
    
    # fourth layer
    c4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(mc32)
    c42 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(c4)
    mc42 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(c42)
    
    # fifth layer
    c5 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(mc42)
    c52 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(c5)
    
    # five to four
    uc54 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(c52), c42], axis=-1)
    uc54c1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(uc54)
    uc54c2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(uc54c1)
    
    # four to three
    uc43 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(uc54c2), c32], axis=-1)
    uc43c1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(uc43)
    uc43c2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(uc43c1)
    
    # three to two
    uc32 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(uc43c2), c22], axis=-1)
    uc32c1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(uc32)
    uc32c2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(uc32c1)
    
    # two to onw
    uc21 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(uc32c2), c12], axis=-1)
    uc21c1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(uc21)
    uc21c2 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(uc21c1)
    logits = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', padding='same')(uc21c2)
    
    with tf.name_scope("DICE"):
        Dice = tl.cost.dice_coe(logits, y)
#%%
    learning_rate = 2e-4
    
    # prob = tf.nn.sigmoid(Lf, name='prob')
    with tf.name_scope("loss"):
        flat_logits = tf.reshape(tensor=logits, shape=(-1, 1))
        flat_labels = tf.reshape(tensor=y, shape=(-1, 1))
    #    cross_entropies = tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
        cross_entropies = tf.keras.losses.binary_crossentropy(flat_labels, flat_logits)
        loss = tf.reduce_mean(cross_entropies, name="loss")
    #    loss_summary = tf.summary.scalar('log_loss', loss)
    
    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(1-Dice)
    return [training_op, loss, Dice, keep_prob, logits, x, y]