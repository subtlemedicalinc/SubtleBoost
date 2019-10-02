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

import keras.callbacks
from keras.optimizers import Adam

from subtle.dnn.generators import GeneratorUNet2D, GeneratorMultiRes2D
from subtle.dnn.adversaries import AdversaryPatch2D
from subtle.dnn.helpers import gan_model, clear_keras_memory, set_keras_memory, load_model, load_data_loader

import subtle.subtle_io as suio
from subtle.data_loaders import SliceLoader
import subtle.subtle_loss as suloss
import subtle.subtle_plot as suplot
import subtle.subtle_args as sargs

usage_str = 'usage: %(prog)s [options]'
description_str = 'Train SubtleGrad network on pre-processed data.'

# FIXME: add time stamps, logging
# FIXME: data augmentation

def save_img(img, fname):
    import matplotlib.pyplot as plt
    plt.set_cmap('gray')

    plt.imshow(img)
    plt.colorbar()
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
    print(args.debug_print())
    print('------\n\n\n')

    if args.max_data_sets is None:
        max_data_sets = np.inf
    else:
        max_data_sets = args.max_data_sets

    if args.gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    np.random.seed(args.random_seed)

    log_tb_dir = os.path.join(args.log_dir, '{}_{}'.format(args.job_id, time.time()))

    # load data
    if args.verbose:
        print('loading data from {}'.format(args.data_list_file))
    tic = time.time()

    case_nums = suio.get_experiment_data(args.experiment, dataset='train')
    data_list = ['{}/{}.h5'.format(args.data_dir, cnum) for cnum in case_nums]

    # each element of the data_list contains 3 sets of 3D
    # volumes containing zero, low, and full contrast.
    # the number of slices may differ but the image dimensions
    # should be the same

    # randomly grab max_data_sets from total data pool
    _ridx = np.random.permutation(len(data_list))
    data_list = [data_list[i] for i in _ridx[:args.max_data_sets]]

    # get dimensions from first file
    if args.gen_type == 'legacy':
        data_shape = suio.get_shape(data_list[0])
        _, _, nx, ny = data_shape
    elif args.gen_type == 'split':
        data_shape = suio.get_shape(data_list[0], params={'h5_key': 'data/X'})
        print(data_shape)
    #FIXME: check that image sizes are the same
        _, nx, ny, nz = data_shape

    clear_keras_memory()
    set_keras_memory(args.keras_memory)

    loss_function = suloss.mixed_loss(l1_lambda=args.l1_lambda, ssim_lambda=args.ssim_lambda, perceptual_lambda=args.perceptual_lambda, wloss_lambda=args.wloss_lambda, img_shape=(nx, ny, 3))
    metrics_monitor = [suloss.l1_loss, suloss.ssim_loss, suloss.mse_loss, suloss.psnr_loss]

    if args.resample_size is not None:
        nx = args.resample_size
        ny = args.resample_size

    if os.path.isfile(args.checkpoint):
        print('Using existing checkpoint at {}'.format(args.checkpoint))
    else:
        print('Creating new checkpoint at {}'.format(args.checkpoint))

    compile_model = (not args.gan_mode)

    model_class = load_model(args.model_name)

    model_kwargs = {
        'num_channel_output': len(args.output_idx),
        'num_channel_first': args.num_channel_first,
        'num_poolings': args.num_poolings,
        'loss_function': loss_function,
        'metrics_monitor': metrics_monitor,
        'lr_init': args.lr_init,
        'batch_norm': args.batch_norm,
        'verbose': args.verbose,
        'checkpoint_file': args.checkpoint,
        'log_dir': log_tb_dir,
        'job_id': args.job_id,
        'save_best_only': args.save_best_only,
        'compile_model': compile_model
    }

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

    model_kwargs = {**model_kwargs, **kw}

    m = model_class(**model_kwargs)
    m.load_weights()

    tic = time.time()

    if len(data_list) == 1 or args.validation_split == 0:
        r = 0
    else: # len(data_list) > 1
        r = int(len(data_list) * args.validation_split)

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
    ckp_monitor = 'model_1_loss' if args.gan_mode else 'val_l1_loss'
    callbacks.append(m.callback_checkpoint(monitor=ckp_monitor))
    callbacks.append(m.callback_tensorbaord(log_dir='{}_plot'.format(log_tb_dir)))

    slice_axis = [0]
    num_epochs = args.num_epochs

    if args.train_mpr:
        slice_axis = [0, 2, 3]
        num_epochs = args.num_epochs // len(slice_axis)
        print('Training in MPR mode: {} epochs'.format(num_epochs))

    if r > 0:
        callbacks.append(m.callback_tbimage(data_list=data_val_list, slice_dict_list=None, slices_per_epoch=1, slices_per_input=args.slices_per_input, batch_size=args.tbimage_batch_size, verbose=args.verbose, residual_mode=args.residual_mode, tag='Validation', gen_type=args.gen_type, log_dir='{}_image'.format(log_tb_dir), shuffle=True, input_idx=args.input_idx, output_idx=args.output_idx, slice_axis=slice_axis, resize=args.resize, resample_size=args.resample_size, brain_only=args.brain_only, brain_only_mode=args.brain_only_mode, model_name=args.model_name, block_size=args.block_size, block_strides=args.block_strides))
    #cb_tensorboard = m.callback_tensorbaord(log_every=1)

    data_loader = load_data_loader(args.model_name)

    gen_kwargs = {
        'data_list': data_train_list,
        'batch_size': args.batch_size,
        'shuffle': args.shuffle,
        'verbose': args.verbose,
        'brain_only': args.brain_only,
        'brain_only_mode': args.brain_only_mode
    }

    if '3d' in args.model_name:
        gen_kw = {
            'block_size': args.block_size,
            'block_strides': args.block_strides
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
        }

    gen_kwargs = {**gen_kwargs, **gen_kw}
    training_generator = data_loader(**gen_kwargs)

    if r > 0:
        gen_kwargs['data_list'] = data_val_list
        gen_kwargs['shuffle'] = False
        validation_generator = data_loader(**gen_kwargs)
    else:
        validation_generator = None

    if args.gan_mode:
        ### TEMP CODE
        fpath_h5 = '/home/srivathsa/projects/studies/gad/tiantan/preprocess/data/NO29.h5'

        pred_gen = SliceLoader(
            data_list=[fpath_h5],
            batch_size=8,
            shuffle=False,
            verbose=0,
            residual_mode=False,
            slices_per_input=7,
            slice_axis=[0]
        )

        ### TEMP CODE

        history_objects = []
        gen = m.model
        adv_model = load_model(args.adversary_name)
        disc_m = adv_model(
            img_rows=nx, img_cols=ny,
            compile_model=compile_model
        )
        disc = disc_m.model

        gan = gan_model(gen, disc, (nx, ny, args.slices_per_input * 2))

        disc.trainable = True
        disc_m._compile_model()
        disc.trainable = False

        gan.compile(loss=[loss_function, 'binary_crossentropy'], optimizer=Adam())
        disc.trainable = True

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

        dc = 30
        real_val = np.ones((X_val.shape[0], dc, dc, 1))

        real = np.ones((args.batch_size, dc, dc, 1))
        fake = np.zeros((args.batch_size, dc, dc, 1))
        real_full = np.ones((data_len * args.batch_size, dc, dc, 1))

        for epoch in range(num_epochs):
            print('EPOCH #{}/{}'.format(epoch + 1, num_epochs))
            indices = np.random.permutation(data_len)

            X_batch = []
            Y_batch = []

            for idx in tqdm(indices, total=data_len):
                X, Y = training_generator.__getitem__(idx)
                gen_imgs = gen.predict(X, batch_size=args.batch_size)

                X_batch.extend(X)
                Y_batch.extend(Y)

                gauss1 = np.random.normal(0, 0.1, Y.shape)
                Y_n = Y + gauss1

                gauss2 = np.random.normal(0, 0.1, gen_imgs.shape)
                g_n = gen_imgs + gauss2

                dis_inp = np.concatenate([Y_n, g_n])
                dis_out = np.concatenate([real, fake])

                history_objects.append(
                    disc.fit(
                        dis_inp, dis_out,
                        batch_size=args.batch_size,
                        epochs=1,
                        verbose=1,
                        #callbacks=callbacks[:-1],
                        shuffle=False

                    )
                )

            X_batch = np.array(X_batch)
            Y_batch = np.array(Y_batch)

            disc.trainable = False
            history_objects.append(
                gan.fit(
                    X_batch, [Y_batch, real_full],
                    epochs=1,
                    batch_size=args.batch_size,
                    verbose=1,
                    callbacks=callbacks[:-1]
                )
            )
            disc.trainable = True

            print('Evaluating...')
            history_objects.append(
                gan.evaluate(
                    X_val, [Y_val, real_val],
                    batch_size=args.batch_size,
                    verbose=1
                )
            )

            epoch_loss = history_objects[-1]
            print('Validation losses:\n')
            print('loss: {}'.format(epoch_loss[0]))
            print('model_1_loss: {}'.format(epoch_loss[1]))
            print('model_2_loss: {}'.format(epoch_loss[2]))

            y_pred = gen.predict_generator(pred_gen)
            pidx = y_pred.shape[0] // 2
            print('saving prediction...')
            save_img(y_pred[pidx, ..., 0], 'epoch_{}'.format(epoch + 1))

            print('End of EPOCH #{}'.format(epoch + 1))

        d_losses = []
        g_losses = []
        for hist in history_objects:
            if isinstance(hist, list):
                g_losses.append(hist[0])
            elif len(hist.history.keys()) == 1:
                d_losses.append(hist.history['loss'][0])
        d_losses = np.clip(d_losses, 0, 1)
        g_losses = np.clip(g_losses, 0, 2)
        plot_losses(d_losses, 'd_losses')
        plot_losses(g_losses, 'g_losses')

    else:
        history = m.model.fit_generator(generator=training_generator, validation_data=validation_generator, validation_steps=args.val_steps_per_epoch, use_multiprocessing=args.use_multiprocessing, workers=args.num_workers, max_queue_size=args.max_queue_size, epochs=num_epochs, steps_per_epoch=args.steps_per_epoch, callbacks=callbacks, verbose=args.verbose, initial_epoch=0)

    toc = time.time()
    print('done training ({:.0f} sec)'.format(toc - tic))

    if args.history_file is not None:
        np.save(args.history_file, history.history)
