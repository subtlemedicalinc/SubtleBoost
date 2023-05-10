#!/usr/bin/env python
'''
inference.py

Inference for contrast synthesis.
Runs the full inference pipeline on a patient

@author: Srivathsa Pasumarthi (srivathsa@subtlemedical.com)
Copyright Subtle Medical (https://www.subtlemedical.com)
Created on 2023/05/04
'''

import tempfile
import os
import datetime
import traceback
import time
import copy
import random
from warnings import warn
import configargparse as argparse
from tqdm import tqdm
from glob import glob
from multiprocessing import Pool, Queue
from scipy.ndimage.interpolation import rotate
import sigpy as sp

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

import numpy as np

from subtle.dnn.helpers import load_model
import subtle.utils.io as utils_io
import subtle.utils.experiment as utils_exp
from subtle.data_loaders import InferenceLoader
import subtle.subtle_metrics as sumetrics
from preprocess import preprocess_chain
import subtle.subtle_preprocess as supre

class ProcArgs:
    def __init__(self, config_dict):
        for key, val in config_dict.items():
            setattr(self, key, val)

def process_mpr(proc_params):
    global gpu_pool
    gpu_id = gpu_pool.get(block=True)

    try:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

        reshape_for_mpr_rotate = proc_params['reshape_for_mpr_rotate']
        slices_per_input = int(proc_params['slices_per_input'])
        input_idx = [int(idx) for idx in proc_params['input_idx'].split(',')]
        num_channel_output = proc_params['num_channel_output']
        num_rotations = proc_params['num_rotations']
        checkpoint = proc_params['checkpoint']
        model_name = proc_params['model_name']
        batch_size = proc_params['batch_size']

        tmpdir = proc_params['tmpdir']
        angle = proc_params['angle']
        slice_axis = proc_params['slice_axis']

        data = np.load('{}/data.npy'.format(tmpdir))
        data = np.clip(data, 0, data.max())

        _, ns, nx, ny = data.shape
        if num_rotations > 1 and angle > 0:
            data_rot = rotate(data, angle, reshape=reshape_for_mpr_rotate, axes=(1, 2))
        else:
            data_rot = data

        proc_args = ProcArgs({'model_name': model_name})
        model_class = load_model(proc_args)

        net = model_class(
            num_channel_input=slices_per_input * len(input_idx),
            num_channel_output=num_channel_output
        ).to('cuda')

        state_dict = torch.load(checkpoint, map_location='cpu')
        net.load_state_dict(state_dict['G'])
        net.eval()

        inf_loader = InferenceLoader(
            input_data=data_rot, slices_per_input=slices_per_input, batch_size=batch_size,
            slice_axis=slice_axis, data_order='stack', resize=(nx, ny)
        )
        num_batches = inf_loader.__len__()
        Y_pred = []
        for idx in tqdm(np.arange(num_batches), total=num_batches):
            X = inf_loader.__getitem__(idx)
            X = torch.from_numpy(X.astype(np.float32)).to('cuda')
            Y = net(X).detach().cpu().numpy()
            Y_pred.extend(Y[:, 0])

        Y_pred = np.array(Y_pred)
        Y_pred = Y_pred[:inf_loader.num_slices, ...] # get rid of slice excess

        if slice_axis == 0:
            pass
        elif slice_axis == 2:
            Y_pred = np.transpose(Y_pred, (1, 0, 2))
        elif slice_axis == 3:
            Y_pred = np.transpose(Y_pred, (1, 2, 0))

        if num_rotations > 1 and angle > 0:
            Y_pred = rotate(
                Y_pred, -angle, reshape=False, axes=(0, 1)
            )

        Y_pred = sp.util.resize(Y_pred, (ns, nx, ny))
        # np.save(
        #     '/home/srivathsa/projects/studies/gad/all/inference/test/pred_{}_{}.npy'.format(
        #         int(angle), slice_axis
        #     ), Y_pred
        # )
        return Y_pred

    except Exception as e:
        print('Exception in thread', e)
        traceback.print_exc()
        return []

    finally:
        gpu_pool.put(gpu_id)

def init_gpu_pool(local_gpu_q):
    global gpu_pool
    gpu_pool = local_gpu_q

if __name__ == '__main__':
    t1 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dcm_pre', type=str, help='DICOM directory of pre-contrast')
    parser.add_argument(
        '--dcm_post', type=str,
        default='DICOM directory of post-contrast (low-dose or standard dose)'
    )

    parser.add_argument(
        '--checkpoint', type=str, help='Path of trained model checkpoint to run inference on'
    )
    parser.add_argument(
        '--config', type=str, help='Path to inference config file',
        default='configs/inference/unified_mpr.json'
    )
    parser.add_argument('--out_folder', type=str, help='DICOM output directory')
    parser.add_argument('--save_npy', action='store_true', default=False)
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument(
        '--vis_dir', type=str, help='Directory to save plot of inference',
        default=None
    )
    parser.add_argument('--gpu', type=str, help='Comma separated GPU IDs', default='0')
    run_args = parser.parse_args()
    mfr = utils_io.get_manufacturer_str(run_args.dcm_pre)

    subexp_name = None
    if 'ge' in mfr.lower():
        subexp_name = 'ge'
    elif 'philips' in mfr.lower():
        subexp_name = 'philips'
    elif 'siemens' in mfr.lower():
        subexp_name = 'siemens'
    else:
        subexp_name = 'other'

    pp_args = utils_exp.get_config(
        fpath_json=run_args.config, subexp_name=subexp_name, config_key='preprocess'
    )
    pp_args.path_zero = run_args.dcm_pre
    pp_args.path_low = run_args.dcm_post
    pp_args.path_full = run_args.dcm_post

    data, data_mask, pp_meta = preprocess_chain(
        pp_args, metadata={'lambda': [], 'inference_only': True}
    )
    data = data.transpose(1, 0, 2, 3)[:2]

    _, ns, nx, ny = data.shape
    inf_args = pp_args = utils_exp.get_config(
        fpath_json=run_args.config, config_key='inference'
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        np.save('{}/data.npy'.format(tmpdir), data)

        if inf_args.inference_mpr:
            slice_axes = [0, 2, 3]
        else:
            slice_axes = [0]

        if inf_args.inference_mpr and inf_args.num_rotations > 1:
            angles = np.linspace(0, 90, inf_args.num_rotations, endpoint=False)
        else:
            angles = [0]

        gpu_ids = inf_args.gpu.split(',')
        gpu_repeat = [[id] * inf_args.procs_per_gpu for id in gpu_ids]
        gpu_ids = [item for sublist in gpu_repeat for item in sublist]
        nworkers = len(gpu_ids)

        gpu_q = Queue()
        for gid in gpu_ids:
            gpu_q.put(gid)

        process_pool = Pool(processes=len(gpu_ids), initializer=init_gpu_pool, initargs=(gpu_q, ))

        proc_params = []
        param_obj = {
            'tmpdir': tmpdir,
            'reshape_for_mpr_rotate': inf_args.reshape_for_mpr_rotate,
            'slices_per_input': inf_args.slices_per_input,
            'input_idx': inf_args.input_idx,
            'num_channel_output': int(inf_args.num_channel_output),
            'num_rotations': int(inf_args.num_rotations),
            'checkpoint': inf_args.checkpoint,
            'model_name': inf_args.model_name,
            'batch_size': inf_args.batch_size
        }

        for angle in angles:
            for ax in slice_axes:
                obj = copy.deepcopy(param_obj)
                obj['angle'] = angle
                obj['slice_axis'] = ax
                proc_params.append(obj)

        proc_results = np.array(process_pool.map(process_mpr, proc_params))
        process_pool.close()
        process_pool.join()

        Y_pred = proc_results.reshape((len(slice_axes), inf_args.num_rotations, ns, nx, ny))
        Y_pred = np.clip(Y_pred, 0, Y_pred.max())

        Y_masks_sum = np.sum(np.array(Y_pred > 0, dtype=np.float), axis=(0, 1), keepdims=False)
        Y_pred = np.divide(
            np.sum(Y_pred, axis=(0, 1), keepdims=False),
            Y_masks_sum,
            where=Y_masks_sum > 0
        )

        rs_size = pp_meta['resampled_size']
        Y_pred = sp.util.resize(Y_pred, (ns, rs_size[0], rs_size[1]))
        Y_pred, _ = supre.zoom_iso(
            Y_pred, np.array(pp_meta['new_spacing']), np.array(pp_meta['old_spacing_low'])
        )

        Y_pred = supre.undo_scaling(Y_pred, pp_meta)
        dpath_save = os.path.join(
            run_args.out_folder, '{}_SubtleGad'.format(run_args.dcm_post.split('/')[-2])
        )
        print('Y_pred final shape', Y_pred.shape)

        utils_io.write_dicoms(
            run_args.dcm_post, Y_pred, dpath_save, series_desc_pre='SubtleGAD pyt: ',
            desc='unified_mpr', series_desc_post='', series_num=inf_args.series_num
        )

        t2 = time.time()
        print('Inference done in {:.3f}secs'.format(t2-t1))
