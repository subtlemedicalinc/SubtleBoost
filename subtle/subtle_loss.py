'''
subtle_loss.py

Custom loss functions for training

@author: Jon Tamir (jon@subtlemedical.com)
Modified from SubtleTrain
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2018/09/25
'''

import sys
import warnings
import time
import numpy as np

#from skimage.measure import compare_ssim, compare_psnr

try:
    from tensorflow import log as tf_log
    from tensorflow import constant as tf_constant
    import keras.losses
    from keras import backend as K
except:
    warnings.warn('import keras failed')

# for extract patches
try:
    from keras_contrib.backend import extract_image_patches as subtle_extract_image_patches
    bypass_ssim_loss = False
except:
    warnings.warn('import keras_contrib failed, replacing ssim loss with L1 loss')
    bypass_ssim_loss = True

from keras.applications.vgg19 import VGG19, preprocess_input as vgg_preprocess
from keras.models import Model

def ssim_loss(y_true, y_pred, kernel=(3, 3), k1=.01, k2=.03, kernel_size=3, max_value=1.):
    # bypass with a zero loss
    if bypass_ssim_loss:
        return keras.losses.mean_absolute_error(y_true, y_true)

    # ssim parameters
    cc1 = (k1 * max_value) ** 2
    cc2 = (k2 * max_value) ** 2

    # extract patches
    y_true = K.reshape(y_true, [-1] + list(K.int_shape(y_pred)[1:]))
    y_pred = K.reshape(y_pred, [-1] + list(K.int_shape(y_pred)[1:]))

    patches_true = subtle_extract_image_patches(y_true, kernel, kernel, 'valid', K.image_data_format())
    patches_pred = subtle_extract_image_patches(y_pred, kernel, kernel, 'valid', K.image_data_format())

    bs, w, h, c1, c2, c3 = K.int_shape(patches_pred)

    patches_true = K.reshape(patches_true, [-1, w, h, c1 * c2 * c3])
    patches_pred = K.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])

    # Get mean
    u_true = K.mean(patches_true, axis=-1)
    u_pred = K.mean(patches_pred, axis=-1)

    # Get variance
    var_true = K.var(patches_true, axis=-1)
    var_pred = K.var(patches_pred, axis=-1)

    # Get covariance
    covar_true_pred = K.mean(patches_true * patches_pred, axis=-1) - u_true * u_pred

    # compute ssim and dssim
    ssim = (2 * u_true * u_pred + cc1) * (2 * covar_true_pred + cc2)
    denom = (K.square(u_true) + K.square(u_pred) + cc1) * (var_pred + var_true + cc2)
    ssim /= denom

    return K.mean((1.0 - ssim) / 2.0)

def mse_loss(y_true, y_pred):
    return keras.losses.mean_squared_error(y_true, y_pred)

def psnr_loss(y_true, y_pred):
    denominator = tf_log(tf_constant(10.0))
    return 20.*tf_log(K.max(y_true)) / denominator - 10. * tf_log(K.mean(K.square(y_pred - y_true))) / denominator

def l1_loss(y_true, y_pred):
    return keras.losses.mean_absolute_error(y_true, y_pred)

def perceptual_loss(y_true, y_pred, img_shape):
    # From https://bit.ly/2HTb4t9

    y_true_3c = K.concatenate([y_true, y_true, y_true])
    y_pred_3c = K.concatenate([y_pred, y_pred, y_pred])

    y_true_3c = vgg_preprocess(y_true_3c)
    y_pred_3c = vgg_preprocess(y_pred_3c)

    vgg = VGG19(include_top=False, weights='imagenet', input_shape=img_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true_3c) - loss_model(y_pred_3c)))

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def mixed_loss(l1_lambda=0.5, ssim_lambda=0.5, perceptual_lambda=0.0, wloss_lambda=0.0, img_shape=(240, 240, 3)):
    if perceptual_lambda > 0 or wloss_lambda > 0:
        return lambda x, y: l1_loss(x, y) * l1_lambda + ssim_loss(x, y) * ssim_lambda + perceptual_loss(x, y, img_shape) * perceptual_lambda + wloss_lambda * wasserstein_loss(x, y)
    return lambda x, y: l1_loss(x, y) * l1_lambda + ssim_loss(x, y) * ssim_lambda
