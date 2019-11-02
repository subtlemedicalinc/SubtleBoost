import os
import json
import random
import time
import warnings
from glob import glob
import gpustat

import numpy as np
import configargparse as argparse

import subtle.subtle_loss as suloss
import subtle.utils.misc as misc_utils
import subtle.utils.hyperparameter as hyp_utils
from subtle.dnn.generators import GeneratorUNet2D
from subtle.data_loaders import SliceLoader
from train import train_process as train_execute


def train_wrap(params, *args):
    ts_hash = misc_utils.get_timestamp_hash(n=4)
    dirpath_trial = os.path.join(params.hyp_log_dir, 'trial_{}'.format(ts_hash))

    if not os.path.exists(dirpath_trial):
        os.makedirs(dirpath_trial)

    params.checkpoint = os.path.join(dirpath_trial, '{}.checkpoint'.format(params.checkpoint_name))
    params.log_tb_dir = os.path.join(dirpath_trial, 'tb')
    params.job_id = ts_hash

    train_execute(params)

def gpu_check(input_gpuids, percent_limit=0.15):
    ip = [int(id) for id in input_gpuids]
    stats = gpustat.GPUStatCollection.new_query().jsonify()

    for gpu_info in stats['gpus']:
        percent_util = float(gpu_info['memory.used'] / gpu_info['memory.total'])
        if (percent_util > percent_limit) and (gpu_info['index'] in ip):
            proc = sorted(gpu_info['processes'], key=lambda d: d['gpu_memory_usage'], reverse=True)[0]

            warnings.warn('Unable to use GPU {}; currently being used by user - "{}"'.format(gpu_info['index'], proc['username']))

            ip.remove(gpu_info['index'])
    if len(ip) == 0:
        raise Exception('None of the GPUs are available. Cannot continue hyperparameter experiment')
    return [str(id) for id in ip]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypsearch_name', type=str, action='store', help='Name of the hyperparameter search')

    args = parser.parse_args()

    if not args.hypsearch_name:
        raise ValueError('Hyperparameter search name should be specified')

    hparams, hyp_config = hyp_utils.get_hypsearch_params(args.hypsearch_name)
    random.seed(hparams.random_seed)
    np.random.seed(hparams.random_seed)

    gpu_ids = gpu_check(hyp_config['gpus'])
    hparams.optimize_parallel_gpu(train_wrap, gpu_ids=gpu_ids, max_nb_trials=hyp_config['trials'])
