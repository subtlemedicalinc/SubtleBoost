#!/usr/bin/env python

import argparse

import numpy as np
from sklearn.model_selection import KFold

usage_str = 'usage: %(prog)s [options] N <input_file> <output_file_train> <output_file_test>'
#description_str = 'pick lines from input and store in output'
description_str = 'split data into KFolds for cross validation'

parser = argparse.ArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('N', action='store', type=int, help='number of folds. test size is len(data)/N')
parser.add_argument('input_file', action='store', type=str, help='input file list')
parser.add_argument('output_file_train', action='store', type=str, help='output file lists will be XXX_<output_file>')
parser.add_argument('output_file_test', action='store', type=str, help='output file lists will be XXX_<output_file>')
#parser.add_argument('output_file', action='store', type=str, help='output file list')
parser.add_argument('--random_seed', action='store', dest='random_seed', type=int, help='random seed', default=None)

args = parser.parse_args()

if args.random_seed is not None:
    np.random.seed(args.random_seed)

f = open(args.input_file)
data_list = f.readlines()
f.close()

L = len(data_list)

kf = KFold(n_splits=args.N, shuffle=True, random_state=args.random_seed)
for idx, (train_index, test_index) in enumerate(kf.split(data_list)):

    output_file_train = '{}_{:03d}'.format(args.output_file_train, idx)
    output_file_test = '{}_{:03d}'.format(args.output_file_test, idx)

    f_train = open(output_file_train, 'w')
    for i in train_index:
        f_train.write(data_list[i])
    f_train.close()

    f_test = open(output_file_test, 'w')
    for i in test_index:
        f_test.write(data_list[i])
    f_test.close()

