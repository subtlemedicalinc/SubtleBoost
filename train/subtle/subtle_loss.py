'''
subtle_loss.py

Custom loss functions for training

@author: Srivathsa Pasumarthi (srivathsa@subtlemedical.com)
Modified from SubtleTrain
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2023/03/14
'''

import sys
import warnings
import time
import numpy as np
from functools import wraps
import math

import torch.nn.functional as F
import torch
from torchvision import models
from torchvision.transforms.functional import resize

# import tensorflow as tf
# from tensorflow import log as tf_log
# from tensorflow import constant as tf_constant
# import keras.losses
# from keras import backend as K
#
# from keras.applications.vgg19 import VGG19
# from keras.models import Model

"""
* Using VGG19's preprocess_input method caused the following error during hyperparameter search execution and hence using mobilenet's method instead

```ValueError: Tensor("loss/linear_model_output_loss/Const_1:0", shape=(3,), dtype=float32) must be from the same graph as Tensor("loss/linear_model_output_loss/strided_slice_6:0", shape=(?, ?, ?, 3), dtype=float32).```

* Imagenet's preprocess_input and vgg19's preprocess input have the same functionality - refer https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py
"""
# from keras.applications.imagenet_utils import preprocess_input as vgg_preprocess


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

        if y_pred.shape[-1] > 1:
            weights = y_true[..., 7:]
            y_true = y_true[..., :7]
        else:
            weights = K.expand_dims(y_true[..., 1])
            y_true = K.expand_dims(y_true[..., 0])

        new_args = [y_true, y_pred, weights]

        if len(args) > 2:
            # perceptual_loss has an extra argument img_shape
            # and ssim_loss has a bunch of extra args
            new_args.extend(args[2:])

        return fn(*new_args, **kwargs)
    return _extract

def extract_image_patches(x, kernel=3, stride=3, dilation=1):
    b,c,h,w = x.shape
    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)
    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    x = F.pad(x, (pad_row//2, pad_row - pad_row//2, pad_col//2, pad_col - pad_col//2))

    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    patches = patches.permute(0,4,5,1,2,3).contiguous()

    return patches #patches.view(b,-1,patches.shape[-2], patches.shape[-1])

def ssim_loss(y_true, y_pred, k1=.01, k2=.03, max_value=1.):
    # ssim parameters
    cc1 = (k1 * max_value) ** 2
    cc2 = (k2 * max_value) ** 2

    patches_true = extract_image_patches(y_true)
    patches_pred = extract_image_patches(y_pred)

    bs, c1, c2, c3, w, h = patches_pred.shape
    patches_true = torch.reshape(patches_true, [-1, c1*c2*c3, w, h])
    patches_pred = torch.reshape(patches_pred, [-1, c1*c2*c3, w, h])

    # Get mean
    u_true = torch.mean(patches_true, dim=1)
    u_pred = torch.mean(patches_pred, dim=1)

    # Get variance
    var_true = torch.var(patches_true, dim=1, unbiased=False)
    var_pred = torch.var(patches_pred, dim=1, unbiased=False)

    # Get covariance
    covar_true_pred = torch.mean(patches_true * patches_pred, dim=1) - (u_true * u_pred)

    # compute ssim and dssim
    ssim = (2 * u_true * u_pred + cc1) * (2 * covar_true_pred + cc2)

    denom = (torch.square(u_true) + torch.square(u_pred) + cc1) * (var_pred + var_true + cc2)
    ssim /= denom
    return torch.mean((1.0 - ssim) / 2.0)

@extract_weights
def mse_loss(y_true, y_pred, weights):
    return keras.losses.mean_squared_error(y_true, y_pred)

@extract_weights
def psnr_loss(y_true, y_pred, weights):
    denominator = tf_log(tf_constant(10.0))
    return 20. * tf_log(K.max(y_true)) / denominator - 10. * tf_log(K.mean(K.square(y_pred - y_true))) / denominator

@extract_weights
def weighted_l1_loss(y_true, y_pred, weights):
    y_true *= weights
    y_pred *= weights
    return keras.losses.mean_absolute_error(y_true, y_pred)

def l1_loss(y_true, y_pred):
    return torch.nn.L1Loss(reduction='none')(y_true, y_pred)

@extract_weights
def perceptual_loss(y_true, y_pred, weights, img_shape, resize_shape):
    # From https://bit.ly/2HTb4t9

    num_slices = int(y_pred.shape[-1])

    vgg = VGG19(include_top=False, weights='imagenet', input_shape=img_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False

    loss_vals = []

    for idx in range(num_slices):
        y_true_sl = K.expand_dims(y_true[..., idx])
        y_pred_sl = K.expand_dims(y_pred[..., idx])

        if resize_shape > 0:
            # For 512x512 images, VGG-19 creates some grid artifacts because the
            # original network is trained with 224x224 images
            y_true_sl = tf.image.resize(y_true_sl, (resize_shape, resize_shape))
            y_pred_sl = tf.image.resize(y_pred_sl, (resize_shape, resize_shape))

        y_true_3c = K.concatenate([y_true_sl, y_true_sl, y_true_sl])
        y_pred_3c = K.concatenate([y_pred_sl, y_pred_sl, y_pred_sl])

        y_true_3c = vgg_preprocess(y_true_3c)
        y_pred_3c = vgg_preprocess(y_pred_3c)

        mse = K.mean(K.square(loss_model(y_true_3c) - loss_model(y_pred_3c)))
        loss_vals.append(mse)

    n_ch = K.variable(float(num_slices))
    loss_vals = tf.stack(loss_vals)
    return tf.math.reduce_mean(loss_vals)

@extract_weights
def perceptual_loss_multi(y_true, y_pred, weights, img_shape):
    y_true_3c = K.concatenate([y_true, y_true, y_true])
    y_pred_3c = K.concatenate([y_pred, y_pred, y_pred])

    y_true_3c = vgg_preprocess(y_true_3c)
    y_pred_3c = vgg_preprocess(y_pred_3c)

    layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv2']

    total_loss = K.variable(0.)

    for lname in layer_names:
        vgg = VGG19(include_top=False, weights='imagenet', input_shape=img_shape)
        loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer(lname).output)
        loss_model.trainable = False
        layer_mse = K.mean(K.square(loss_model(y_true_3c) - loss_model(y_pred_3c)))

        total_loss.assign_add(layer_mse)

    return total_loss

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def compute_style_loss(style, combination, img_shape):
    style = gram_matrix(style)
    combination = gram_matrix(combination)
    size = img_shape[0] * img_shape[1]
    return K.sum(K.square(style - combination)) / (4. * (img_shape[2] ** 2) * (size ** 2))

@extract_weights
def style_loss(y_true, y_pred, weights, img_shape):
    # From https://github.com/gsurma/style_transfer/blob/master/style-transfer.ipynb

    y_true_3c = K.concatenate([y_true, y_true, y_true])
    y_true_3c = vgg_preprocess(y_true_3c)

    combi_img = K.placeholder((1, img_shape[0], img_shape[1], 3))
    input_tensor = K.concatenate([y_true_3c, y_true_3c, combi_img], axis=0)

    model = VGG19(input_tensor=input_tensor, include_top=False, weights='imagenet')

    layers = dict([(layer.name, layer.output) for layer in model.layers])
    style_layers = [
        "block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"
    ]

    loss = K.variable(0.)
    for layer_name in style_layers:
        layer_features = layers[layer_name]
        style_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss.assign_add(compute_style_loss(style_features, combination_features, img_shape))

    return loss

@extract_weights
def wasserstein_loss(y_true, y_pred, weights):
    return K.mean(y_true * y_pred)

class VGGLoss(torch.nn.Module):
    def __init__(self, fpath_ckp, requires_grad=False, img_resize=0):
        super(VGGLoss, self).__init__()
        self.fpath_ckp = fpath_ckp
        vgg19 = models.vgg19()

        ckp = torch.load(fpath_ckp, map_location='cpu')
        vgg19.load_state_dict(ckp)

        self.blocks = vgg19.features[:16]

        for bl in self.blocks:
            for p in bl.parameters():
                p.requires_grad = requires_grad

        self.img_resize = img_resize
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to('cuda')
        self.mean.requires_grad = False

        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to('cuda')
        self.std.requires_grad = False

        self._z = torch.tensor(0).to('cuda')

    def vgg_feat(self, x):
        x = x.repeat(1, 3, 1, 1)
        if self.img_resize > 0:
            # print('x before resize', x.min().item(), x.max().item())
            x = resize(x, [self.img_resize, self.img_resize])
            x = torch.clip(x, self._z, x.max())
            # print('x after resize', x.min().item(), x.max().item())

        # x = (x - x.min()) / (x.max() - x.min())
        # x = (x - self.mean) / self.std

        # diff = (x - self.mean)
        # ft = self.blocks(diff.type(torch.cuda.FloatTensor))
        ft = self.blocks(x.type(torch.cuda.FloatTensor))
        return ft

    def forward(self, y_true, y_pred):
        y_true = y_true[:, 0, ...]
        loss_vals = 0

        for sl in np.arange(y_true.shape[0]):
            ft1 = self.vgg_feat(y_true[sl][None])
            ft2 = self.vgg_feat(y_pred[sl][None])

            mse = torch.mean(torch.square(ft1 - ft2))
            loss_vals += mse

        return loss_vals / y_true.shape[0]


def mixed_loss(args, y_true, y_pred, vgg_loss):
    total_loss = 0
    indiv_loss = {}

    if args.l1_lambda > 0:
        if args.enh_mask:
            y_t = y_true[:, 0]
            weights = y_true[:, 1]

            y_t = y_t * weights
            y_p = y_pred[:, 0] * weights

            l1 = l1_loss(y_t, y_p)
        else:
            l1 = l1_loss(y_true[:, 0], y_pred[:, 0])
        total_loss += args.l1_lambda * l1
        indiv_loss['l1'] = l1

    if args.ssim_lambda > 0:
        ssim = ssim_loss(y_true[:, 0][:, None], y_pred)
        total_loss += args.ssim_lambda * ssim
        indiv_loss['ssim'] = ssim
    else:
        indiv_loss['ssim'] = l1

    if vgg_loss is not None:
        ploss = vgg_loss(y_true, y_pred)
        total_loss += args.perceptual_lambda * ploss
        indiv_loss['perceptual'] = ploss

    return total_loss, indiv_loss
