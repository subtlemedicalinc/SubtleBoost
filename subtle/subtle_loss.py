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
from functools import wraps

import tensorflow as tf
from tensorflow import log as tf_log
from tensorflow import constant as tf_constant
import keras.losses
from keras import backend as K

from keras.applications.vgg19 import VGG19
from keras.models import Model

"""
* Using VGG19's preprocess_input method caused the following error during hyperparameter search execution and hence using mobilenet's method instead

```ValueError: Tensor("loss/linear_model_output_loss/Const_1:0", shape=(3,), dtype=float32) must be from the same graph as Tensor("loss/linear_model_output_loss/strided_slice_6:0", shape=(?, ?, ?, 3), dtype=float32).```

* Imagenet's preprocess_input and vgg19's preprocess input have the same functionality - refer https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py
"""
from keras.applications.imagenet_utils import preprocess_input as vgg_preprocess


"""
Decorator for splitting y_true into ground truth image and mask weights
"""
def extract_weights(fn):
    @wraps(fn)
    # This is to say that this decorator is wrapping the given function so that
    # the __name__ attribute is meaningful and looks right in keras progressbar

    def _extract(*args, **kwargs):
        y_true = args[0]
        y_pred = args[1]

        weights = K.expand_dims(y_true[..., 1])
        y_true = K.expand_dims(y_true[..., 0])

        new_args = [y_true, y_pred, weights]

        if len(args) > 2:
            # perceptual_loss has an extra argument img_shape
            # and ssim_loss has a bunch of extra args
            new_args.extend(args[2:])

        return fn(*new_args, **kwargs)
    return _extract

"""
Based on implementation from https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/backend/tensorflow_backend.py

Modified by Srivathsa Pasumarthi for 3D patches
"""
def extract_image_patches(x, ksizes, ssizes, padding='same',
                          data_format='channels_last'):
    """Extract the patches from an image.
    # Arguments
        x: The input image
        ksizes: 2-d tuple with the kernel size
        ssizes: 2-d tuple with the strides size
        padding: 'same' or 'valid'
        data_format: 'channels_last' or 'channels_first'
    # Returns
        The (k_w,k_h) patches extracted
        TF ==> (batch_size,w,h,k_w,k_h,c)
        TH ==> (batch_size,w,h,c,k_w,k_h)
    """

    if K.ndim(x) == 5:
        bs_i, d_i, w_i, h_i, ch_i = K.int_shape(x)
        kernel = [1] + [ksizes[0]] * 3 + [1]
        strides = [1] + [ssizes[0]] * 3 + [1]

        patches = tf.extract_volume_patches(x, kernel, strides, padding)
        bs, d, w, h, ch = K.int_shape(patches)
        reshaped = tf.reshape(patches, [-1, d, w, h, tf.floordiv(ch, ch_i), ch_i])
        final_shape = [-1, d, w, h, ch_i, ksizes[0], ksizes[0], ksizes[0]]
        patches = tf.reshape(tf.transpose(reshaped, [0, 1, 2, 3, 5, 4]), final_shape)

        patches = K.permute_dimensions(patches, [0, 1, 2, 3, 5, 6, 7, 4])
    elif K.ndim(x) == 4:
        bs_i, w_i, h_i, ch_i = K.int_shape(x)
        kernel = [1, ksizes[0], ksizes[1], 1]
        strides = [1, ssizes[0], ssizes[1], 1]

        patches = tf.extract_image_patches(x, kernel, strides, [1, 1, 1, 1], padding)
        bs, w, h, ch = K.int_shape(patches)
        reshaped = tf.reshape(patches, [-1, w, h, tf.floordiv(ch, ch_i), ch_i])
        final_shape = [-1, w, h, ch_i, ksizes[0], ksizes[1]]
        patches = tf.reshape(tf.transpose(reshaped, [0, 1, 2, 4, 3]), final_shape)

        patches = K.permute_dimensions(patches, [0, 1, 2, 4, 5, 3])

    return patches

@extract_weights
def ssim_loss(y_true, y_pred, weights, kernel=(3, 3), k1=.01, k2=.03, kernel_size=3, max_value=1.):
    # ssim parameters
    cc1 = (k1 * max_value) ** 2
    cc2 = (k2 * max_value) ** 2

    # extract patches
    y_true = K.reshape(y_true, [-1] + list(K.int_shape(y_pred)[1:]))
    y_pred = K.reshape(y_pred, [-1] + list(K.int_shape(y_pred)[1:]))

    patches_true = extract_image_patches(y_true, kernel, kernel, 'VALID', K.image_data_format())
    patches_pred = extract_image_patches(y_pred, kernel, kernel, 'VALID', K.image_data_format())

    if K.ndim(y_true) == 4:
        bs, w, h, c1, c2, c3 = K.int_shape(patches_pred)
        patches_true = K.reshape(patches_true, [-1, w, h, c1 * c2 * c3])
        patches_pred = K.reshape(patches_pred, [-1, w, h, c1 * c2 * c3])
    elif K.ndim(y_true) == 5:
        bs, d, w, h, c1, c2, c3, c4 = K.int_shape(patches_pred)
        patches_true = K.reshape(patches_true, [-1, d, w, h, c1 * c2 * c3 * c4])
        patches_pred = K.reshape(patches_pred, [-1, d, w, h, c1 * c2 * c3 * c4])

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

@extract_weights
def mse_loss(y_true, y_pred, weights):
    return keras.losses.mean_squared_error(y_true, y_pred)

@extract_weights
def psnr_loss(y_true, y_pred, weights):
    denominator = tf_log(tf_constant(10.0))
    return 20.*tf_log(K.max(y_true)) / denominator - 10. * tf_log(K.mean(K.square(y_pred - y_true))) / denominator

@extract_weights
def weighted_l1_loss(y_true, y_pred, weights):
    y_true *= weights
    y_pred *= weights
    return keras.losses.mean_absolute_error(y_true, y_pred)

@extract_weights
def l1_loss(y_true, y_pred, weights):
    return keras.losses.mean_absolute_error(y_true, y_pred)

@extract_weights
def perceptual_loss(y_true, y_pred, weights, img_shape):
    # From https://bit.ly/2HTb4t9

    y_true_3c = K.concatenate([y_true, y_true, y_true])
    y_pred_3c = K.concatenate([y_pred, y_pred, y_pred])

    y_true_3c = vgg_preprocess(y_true_3c)
    y_pred_3c = vgg_preprocess(y_pred_3c)

    vgg = VGG19(include_top=False, weights='imagenet', input_shape=img_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true_3c) - loss_model(y_pred_3c)))

@extract_weights
def wasserstein_loss(y_true, y_pred, weights):
    return K.mean(y_true * y_pred)

def mixed_loss(l1_lambda=0.5, ssim_lambda=0.5, perceptual_lambda=0.0, wloss_lambda=0.0, img_shape=(240, 240, 3), enh_mask=False):
    l1_fn = l1_loss if not enh_mask else weighted_l1_loss

    if perceptual_lambda > 0 or wloss_lambda > 0:
        return lambda x, y: l1_fn(x, y) * l1_lambda + ssim_loss(x, y) * ssim_lambda + perceptual_loss(x, y, img_shape) * perceptual_lambda + wloss_lambda * wasserstein_loss(x, y)
    return lambda x, y: l1_fn(x, y) * l1_lambda + ssim_loss(x, y) * ssim_lambda
