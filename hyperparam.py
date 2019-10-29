import os
import json
import random
import time
from glob import glob

import numpy as np
import configargparse as argparse
from test_tube import HyperOptArgumentParser

import subtle.subtle_loss as suloss
import subtle.subtle_io as suio
from subtle.dnn.generators import GeneratorUNet2D
from subtle.data_loaders import SliceLoader
from train import train_process as train_execute

def train_wrap(params, *args):
    ts_hash = suio.get_timestamp_hash(n=4)
    dirpath_trial = os.path.join(params.hyp_log_dir, 'trial_{}'.format(ts_hash))

    if not os.path.exists(dirpath_trial):
        os.makedirs(dirpath_trial)

    params.checkpoint = os.path.join(dirpath_trial, '{}.checkpoint'.format(params.checkpoint_name))
    params.log_tb_dir = os.path.join(dirpath_trial, 'tb')
    params.job_id = ts_hash

    train_execute(params)

if __name__ == '__main__':
    seed_val = 12321
    random.seed(seed_val)
    np.random.seed(seed_val)

    parser = argparse.ArgumentParser()
    parser.add_argument('--hypsearch_name', type=str, action='store', help='Name of the hyperparameter search')

    args = parser.parse_args()

    if not args.hypsearch_name:
        raise ValueError('Hyperparameter search name should be specified')

    hparams, hyp_config = suio.get_hypsearch_params(args.hypsearch_name)
    hparams.optimize_parallel_gpu(train_wrap, gpu_ids=hyp_config['gpus'], max_nb_trials=hyp_config['trials'])
