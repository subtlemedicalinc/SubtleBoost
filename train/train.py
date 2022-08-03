#!/usr/bin/env python
'''
train.py

Training for contrast synthesis.
Trains netowrk using dataset of npy files

@author: Jon Tamir (jon@subtlemedical.com)
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2018/05/25
'''

import os
import datetime
import time
import random
from warnings import warn
import configargparse as argparse
from tqdm import tqdm

import numpy as np

from keras.callbacks import TensorBoard
from keras.optimizers import Adam

from subtle.dnn.generators import GeneratorUNet2D, GeneratorMultiRes2D
from subtle.dnn.adversaries import AdversaryPatch2D
from subtle.dnn.helpers import gan_model, clear_keras_memory, set_keras_memory, load_model, load_data_loader

import subtle.utils.io as utils_io
import subtle.utils.experiment as utils_exp
import subtle.utils.hyperparameter as utils_hyp
from subtle.data_loaders import SliceLoader
import subtle.subtle_loss as suloss
import subtle.subtle_metrics as sumetrics
import subtle.subtle_plot as suplot
from subtle.dnn.callbacks import plot_tb

usage_str = 'usage: %(prog)s [options]'
description_str = 'Train SubtleGrad network on pre-processed data.'

# FIXME: add time stamps, logging
# FIXME: data augmentation

def save_img(img, fname, title=None):
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = (20, 8)

    plt.set_cmap('gray')

    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.savefig('/home/srivathsa/projects/studies/gad/tiantan/train/logs/test/{}.png'.format(fname))
    plt.clf()

def plot_losses(losses, fname):
    import matplotlib.pyplot as plt
    plt.set_cmap('gray')

    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('/home/srivathsa/projects/studies/gad/tiantan/train/logs/test/{}.png'.format(fname))
    plt.clf()

def train_process(args):
    print('------')
    print(args)
    print('------\n\n\n')

    try:
        hypsearch = (args.hypsearch_name is not None)
    except Exception:
        hypsearch = False

    if args.max_data_sets is None:
        max_data_sets = np.inf
    else:
        max_data_sets = args.max_data_sets

    if args.gpu is not None and not hypsearch:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    np.random.seed(args.random_seed)

    try:
        log_tb_dir = args.log_tb_dir
    except Exception:
        log_tb_dir = os.path.join(args.log_dir, '{}_{}'.format(args.job_id, time.time()))

    # load data
    if args.verbose:
        print('loading data from {}'.format(args.data_list_file))
    tic = time.time()

    case_nums = utils_exp.get_experiment_data(args.experiment, dataset='train')
    data_list = ['{}/{}.{}'.format(args.data_dir, cnum, args.file_ext) for cnum in case_nums]

    # each element of the data_list contains 3 sets of 3D
    # volumes containing zero, low, and full contrast.
    # the number of slices may differ but the image dimensions
    # should be the same

    # randomly grab max_data_sets from total data pool
    _ridx = np.random.permutation(len(data_list))
    data_list = [data_list[i] for i in _ridx[:args.max_data_sets]]

    # get dimensions from first file
    if args.gen_type == 'legacy':
        data_shape = utils_io.get_shape(data_list[0])
        _, _, nx, ny = data_shape
    elif args.gen_type == 'split':
        data_shape = utils_io.get_shape(data_list[0], params={'h5_key': 'data/X'})
        print(data_shape)
    #FIXME: check that image sizes are the same
        _, nx, ny, nz = data_shape

    args.input_idx = [int(idx) for idx in args.input_idx.split(',')]
    args.output_idx = [int(idx) for idx in args.output_idx.split(',')]

    fpath_uad_masks = []

    if args.enh_mask_uad:
        args.uad_file_ext = (args.uad_file_ext or args.file_ext)
        fpath_uad_masks = [
            '{}/{}.{}'.format(args.uad_mask_path, cnum, args.uad_file_ext)
            for cnum in case_nums
        ]

    enh_mask_mode = (args.enh_mask or args.enh_mask_uad)

    if not hypsearch:
        clear_keras_memory()

    set_keras_memory(args.keras_memory)

    # lw_sum = np.sum([args.l1_lambda, args.ssim_lambda, args.perceptual_lambda, args.wloss_lambda, args.style_lambda])

    # if lw_sum > 1.0:
    #     args.l1_lambda /= lw_sum
    #     args.ssim_lambda /= lw_sum
    #     args.perceptual_lambda /= lw_sum
    #     args.wloss_lambda /= lw_sum
    #
    #     print('Loss weight sum is > 1. Normalizing loss weights to add up to one. New loss weights are: \n\n# l1_lambda={:.3f}\n# ssim_lambda={:.3f}\n# perceptual_lambda={:.3f}\n# wloss_lambda={:.3f}'.format(args.l1_lambda, args.ssim_lambda, args.perceptual_lambda, args.wloss_lambda, args.style_lambda))

    loss_function = suloss.mixed_loss(l1_lambda=args.l1_lambda, ssim_lambda=args.ssim_lambda, perceptual_lambda=args.perceptual_lambda, wloss_lambda=args.wloss_lambda, style_lambda=args.style_lambda, img_shape=(nx, ny, 3), enh_mask=enh_mask_mode, vgg_resize_shape=args.vgg_resize_shape)

    l1_metric = suloss.l1_loss if not enh_mask_mode else suloss.weighted_l1_loss
    metrics_monitor = [l1_metric, suloss.ssim_loss, suloss.mse_loss, suloss.psnr_loss]

    if enh_mask_mode and args.verbose:
        print('Using weighted L1 loss...')

    if args.resample_size is not None:
        nx = args.resample_size
        ny = args.resample_size

    if os.path.isfile(args.checkpoint):
        print('Using existing checkpoint at {}'.format(args.checkpoint))
    else:
        print('Creating new checkpoint at {}'.format(args.checkpoint))

    compile_model = (not args.gan_mode)
    model_class = load_model(args.model_name)

    if args.pretrain_ckps is not None:
        ckp_dir = os.path.dirname(args.checkpoint_dir)
        fpaths_pre = [
            '{}/{}.checkpoint'.format(ckp_dir, ckp)
            for ckp in args.pretrain_ckps.split(',')
        ]
    else:
        fpaths_pre = []

    model_kwargs = {
        'model_config': args.model_config,
        'num_channel_output': len(args.output_idx) if not args.multi_slice_gt else args.slices_per_input,
        'loss_function': loss_function,
        'metrics_monitor': metrics_monitor,
        'lr_init': args.lr_init,
        'verbose': args.verbose,
        'checkpoint_file': args.checkpoint,
        'log_dir': log_tb_dir,
        'job_id': args.job_id,
        'save_best_only': args.save_best_only,
        'compile_model': compile_model,
        'fpaths_pre': fpaths_pre,
        'transfer_weights': True
    }

    if hypsearch:
        plot_list = utils_hyp.get_hyp_plot_list(args.hypsearch_name)

        tunable_exp_params, tunable_model_params = utils_hyp.get_tunable_params(args.hypsearch_name)

        tunable_model_params = {k: args.__dict__['__model_{}'.format(k)] for k in tunable_model_params.keys()}
    else:
        plot_list = utils_exp.get_experiment_data(args.experiment, dataset='plot')
        tunable_exp_params = None
        tunable_model_params = None

    if plot_list is not None:
        plot_list = [
            ('{}/{}.{}'.format(args.data_dir, p[0], args.file_ext), p[1])
            for p in plot_list
        ]

    model_kwargs['tunable_params'] = tunable_model_params

    if '3d' in args.model_name:
        kw = {
            'img_rows': args.block_size,
            'img_cols': args.block_size,
            'img_depth': args.block_size,
            'num_channel_input': 2
        }
    else:
        kw = {
            'img_rows': nx,
            'img_cols': ny,
            'num_channel_input': len(args.input_idx) * args.slices_per_input
        }

    if args.use_uad_ch_input:
        kw['num_channel_input'] += args.uad_ip_channels

    model_kwargs = {**model_kwargs, **kw}

    m = model_class(**model_kwargs)
    m.load_weights()

    tic = time.time()

    if len(data_list) == 1 or args.validation_split == 0:
        r = 0
    else: # len(data_list) > 1
        r = int(len(data_list) * args.validation_split)

    val_data = utils_exp.get_experiment_data(args.experiment, dataset='val')
    if len(val_data) > 0:
        data_val_list = ['{}/{}.{}'.format(args.data_dir, v, args.file_ext) for v in val_data]
        data_train_list = data_list
    else:
        data_val_list = data_list[:r]
        data_train_list = data_list[r:]

    if args.verbose:
        print('using {} datasets for training:'.format(len(data_train_list)))
        for d in data_train_list:
            print(d)
        print('using {} datasets for validation:'.format(len(data_val_list)))
        for d in data_val_list:
            print(d)


    callbacks = []
    ckp_monitor = None

    if args.gan_mode:
        ckp_monitor = 'gen_loss'
    else:
        if enh_mask_mode:
            ckp_monitor = 'val_weighted_l1_loss'
        else:
            ckp_monitor = 'val_l1_loss'

    callbacks.append(m.callback_checkpoint(monitor=ckp_monitor))

    tb_logdir = '{}_plot'.format(log_tb_dir)
    if not args.gan_mode:
        callbacks.append(m.callback_tensorboard(log_dir=tb_logdir))

    slice_axis = [0]
    num_epochs = args.num_epochs

    if args.train_mpr:
        slice_axis = [0, 2, 3]
        # num_epochs = args.num_epochs // len(slice_axis)
        print('Training in MPR mode: {} epochs'.format(num_epochs))

    if r > 0:
        # FIXME: change the tbimage callback to take a generator, so that all this stuff doesn't have to be passed explicitly
        callbacks.append(m.callback_tbimage(data_list=data_val_list, slice_dict_list=None, slices_per_epoch=1, slices_per_input=args.slices_per_input, batch_size=args.tbimage_batch_size, verbose=args.verbose, residual_mode=args.residual_mode, tag='Validation', gen_type=args.gen_type, log_dir='{}_image'.format(log_tb_dir), shuffle=True, input_idx=args.input_idx, output_idx=args.output_idx, slice_axis=slice_axis, resize=args.resize, resample_size=args.resample_size, brain_only=args.brain_only, brain_only_mode=args.brain_only_mode, model_name=args.model_name, block_size=args.block_size, block_strides=args.block_strides, gan_mode=args.gan_mode, use_enh_mask=args.enh_mask, enh_pfactor=args.enh_pfactor, detailed_plot=(not hypsearch), plot_list=plot_list, file_ext=args.file_ext, uad_mask_path=args.uad_mask_path, uad_file_ext=args.uad_file_ext, use_enh_uad=args.enh_mask_uad, use_uad_ch_input=args.use_uad_ch_input, uad_ip_channels=args.uad_ip_channels, uad_mask_threshold=args.uad_mask_threshold, enh_mask_t2=args.enh_mask_t2, multi_slice_gt=args.multi_slice_gt))

    data_loader = load_data_loader(args.model_name)

    gen_kwargs = {
        'data_list': data_train_list,
        'batch_size': args.batch_size,
        'shuffle': args.shuffle,
        'verbose': args.verbose,
        'brain_only': args.brain_only,
        'brain_only_mode': args.brain_only_mode,
        'use_enh_uad': args.enh_mask_uad,
        'use_uad_ch_input': args.use_uad_ch_input,
        'uad_ip_channels': args.uad_ip_channels,
        'fpath_uad_masks': fpath_uad_masks,
        'uad_mask_threshold': args.uad_mask_threshold,
        'uad_mask_path': args.uad_mask_path,
        'uad_file_ext': args.uad_file_ext,
        'enh_mask_t2': args.enh_mask_t2,
        'multi_slice_gt': args.multi_slice_gt
    }

    if '3d' in args.model_name:
        gen_kw = {
            'block_size': args.block_size,
            'block_strides': args.block_strides,
        }
    else:
        gen_kw = {
            'residual_mode': args.residual_mode,
            'positive_only':  args.positive_only,
            'slices_per_input': args.slices_per_input,
            'input_idx': args.input_idx,
            'output_idx': args.output_idx,
            'slice_axis': slice_axis,
            'resize': args.resize,
            'resample_size': args.resample_size,
            'use_enh_mask': args.enh_mask,
            'enh_pfactor': args.enh_pfactor,
            'file_ext': args.file_ext
        }

    gen_kwargs = {**gen_kwargs, **gen_kw}
    training_generator = data_loader(**gen_kwargs)

    if r > 0:
        gen_kwargs['data_list'] = data_val_list
        gen_kwargs['shuffle'] = False
        validation_generator = data_loader(**gen_kwargs)
    else:
        validation_generator = None

    if hypsearch:
        # Hypsearch callback #1 - training progress bar
        callbacks.append(
            m.callback_progress(log_dir=tb_logdir, data_loader=training_generator)
        )

        # Hypsearch callback #2 - display the trial params as a table on tensorboard
        src_dict = {**args.__dict__, **m.config_dict}
        tunable_args = {
            k: src_dict[k]
            for k in {**tunable_exp_params, **tunable_model_params}.keys()
        }

        hparams_log = os.path.join(os.path.dirname(args.checkpoint), 'tb_text')
        callbacks.append(
            m.callback_hparams(log_dir=hparams_log, tunable_args=tunable_args)
        )

        # Hypsearch callback #3 - CSV logger for hypmonitor
        fpath_csv = os.path.join(os.path.dirname(args.checkpoint), 'metrics.csv')
        callbacks.append(m.callback_csv(fpath_csv=fpath_csv))

    if args.gan_mode:
        gen = m.model
        adv_model = load_model(args.adversary_name)
        disc_m = adv_model(
            img_rows=nx, img_cols=ny, compile_model=compile_model,
            lr_init=args.disc_lr_init, beta=args.disc_beta, loss_function=args.disc_loss_function
        )
        disc = disc_m.model

        gan = gan_model(gen, disc, (nx, ny, args.slices_per_input * 2))

        disc.trainable = True
        disc_m._compile_model()
        disc.trainable = False

        gan.compile(loss=[loss_function, args.disc_loss_function], optimizer=Adam())
        disc.trainable = True

        tb_callback = TensorBoard(tb_logdir)
        tb_callback.set_model(gan)
        tb_names = ['train_gloss', 'train_dloss', 'val_gloss', 'val_dloss']

        data_len = training_generator.__len__()
        val_len = validation_generator.__len__()

        print('Pre-fetching validation data')

        X_val = []
        Y_val = []

        for val_idx in tqdm(range(val_len), total=val_len):
            xv, yv = validation_generator.__getitem__(val_idx)
            X_val.extend(xv)
            Y_val.extend(yv)

        X_val = np.array(X_val)
        Y_val = np.array(Y_val)

        real_val = disc_m.get_real_gt(X_val.shape[0])

        real = disc_m.get_real_gt(args.batch_size)
        fake = disc_m.get_fake_gt(args.batch_size)
        real_full = disc_m.get_real_gt(data_len * args.batch_size)

        # for saving epoch output as a npy file
        fpath_h5 = '/home/srivathsa/projects/studies/gad/tiantan/preprocess/data/NO31.h5'

        pred_gen = SliceLoader(
            data_list=[fpath_h5],
            batch_size=1,
            predict=True,
            shuffle=False,
            verbose=0,
            residual_mode=False,
            slices_per_input=args.slices_per_input,
            slice_axis=[0],
            file_ext=args.file_ext
        )

        epoch_preds = []

        for epoch in range(num_epochs):
            print('\nEPOCH #{}/{}'.format(epoch + 1, num_epochs))
            indices = np.random.permutation(data_len)

            X_batch = []
            Y_batch = []

            train_dloss = []
            for idx in tqdm(indices, total=data_len):
                X, Y = training_generator.__getitem__(idx)

                if Y.shape[3] > 1:
                    # enh mask case
                    Y = np.array([Y[..., 0]]).transpose(1, 2, 3, 0)
                gen_imgs = gen.predict(X, batch_size=args.batch_size)

                X_batch.extend(X)
                Y_batch.extend(Y)

                if args.add_disc_noise:
                    gauss1 = np.random.normal(Y.min(), 0.02 * Y.max(), Y.shape)
                    Y_n = Y + gauss1

                    gauss2 = np.random.normal(gen_imgs.min(), 0.02 * gen_imgs.max(), gen_imgs.shape)
                    g_n = gen_imgs + gauss2
                else:
                    Y_n = Y
                    g_n = gen_imgs

                for _ in range(args.num_disc_steps):
                    dhist1 = disc.fit(
                        Y_n, real,
                        batch_size=args.batch_size,
                        epochs=1,
                        verbose=1,
                        shuffle=True
                    )

                    dhist2 = disc.fit(
                        g_n, fake,
                        batch_size=args.batch_size,
                        epochs=1,
                        verbose=1,
                        shuffle=True
                    )

                    dloss1 = dhist1.history['loss'][0]
                    dloss2 = dhist1.history['loss'][0]

                    dloss = 0.5 * np.add(dloss1, dloss2)
                    train_dloss.append(dloss)

            train_dloss = np.mean(train_dloss)

            X_batch = np.array(X_batch)
            Y_batch = np.array(Y_batch)

            disc.trainable = False
            ghist = gan.fit(
                X_batch, [Y_batch, real_full],
                epochs=1,
                batch_size=args.batch_size,
                verbose=1,
                callbacks=callbacks
            )
            disc.trainable = True

            train_gloss = ghist.history['gen_loss'][0]

            epoch_loss = gan.evaluate(
                X_val, [Y_val, real_val],
                batch_size=args.batch_size,
                verbose=1
            )

            val_gloss = epoch_loss[0]
            val_dloss = epoch_loss[2]

            plot_tb(callback=tb_callback, names=tb_names, logs=[train_gloss, train_dloss, val_gloss, val_dloss], batch_no=epoch)

            y_pred = gen.predict_generator(pred_gen)
            pidx = 99 # this slice in NO31 has vasculature inside tumor

            gtruth_pred = utils_io.load_file(fpath_h5)
            out_img = np.hstack([
                gtruth_pred[pidx, 0],
                gtruth_pred[pidx, 1],
                gtruth_pred[pidx, 2],
                y_pred[pidx, ..., 0]
            ])
            out_psnr = sumetrics.psnr(gtruth_pred[:, 2], y_pred[..., 0])
            out_ssim = sumetrics.ssim(gtruth_pred[:, 2], y_pred[..., 0])

            epoch_preds.append((out_img, out_psnr, out_ssim))
            np.save('/home/srivathsa/projects/studies/gad/tiantan/train/logs/gan_progress.npy', np.array(epoch_preds))
    else:
        history = m.model.fit_generator(generator=training_generator, validation_data=validation_generator, validation_steps=args.val_steps_per_epoch, use_multiprocessing=args.use_multiprocessing, workers=args.num_workers, max_queue_size=args.max_queue_size, epochs=num_epochs, steps_per_epoch=args.steps_per_epoch, callbacks=callbacks, verbose=args.verbose, initial_epoch=0)

    toc = time.time()
    print('done training ({:.0f} sec)'.format(toc - tic))

    if args.history_file is not None:
        np.save(args.history_file, history.history)
