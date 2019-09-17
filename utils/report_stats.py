#!/usr/bin/env python

import sys

import h5py
import numpy as np
import argparse
import os.path


usage_str = 'usage: %(prog)s [--key <val>] file1.h5 [file2.h5, ...]'
description_str = 'aggregate stats across datasets'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('inputs', nargs='+', type=str, help='input h5 files')
    parser.add_argument('--key', type=str, action='append', default=None, dest='keys', help='key to parse e.g. pred/nrmse')

    args = parser.parse_args()

    assert args.keys is not None, 'must specify at least one key'

    data = {}
    input_keys = [os.path.basename(_in) for _in in args.inputs]

    for i, input_key in enumerate(input_keys):
        data[input_key] = {}
        with h5py.File(args.inputs[i], 'r') as f:
            for key in args.keys:
                data[input_key][key] = np.array(f[key])

    for key in args.keys:
        print(key, 'mean', 'stdev')
        vals = np.zeros((len(input_keys), 2))
        for i, input_key in enumerate(input_keys):
            mm = np.mean(data[input_key][key])
            ss = np.std(data[input_key][key])
            print(input_key, mm, ss)
            vals[i] = mm, ss
        print('mean of vals:', np.mean(vals, axis=0))
        print('std of vals:', np.std(vals, axis=0))
        print()
