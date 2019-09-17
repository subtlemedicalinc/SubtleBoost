#!/usr/bin/env python

import argparse

import numpy as np

usage_str = 'usage: %(prog)s [options] N <input_file>'
#description_str = 'pick lines from input and store in output'
description_str = 'pick lines from input and write to stdout'

parser = argparse.ArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('N', action='store', type=int, help='number of items to grab')
parser.add_argument('input_file', action='store', type=str, help='input file list')
#parser.add_argument('output_file', action='store', type=str, help='output file list')
parser.add_argument('--random_seed', action='store', dest='random_seed', type=int, help='random seed', default=None)

args = parser.parse_args()

if args.random_seed is not None:
    np.random.seed(args.random_seed)

f = open(args.input_file)
lines = f.readlines()
f.close()

L = len(lines)

ridx = np.random.permutation(L)[:args.N]

lines2 = np.array(lines)[ridx]

for line in lines2:
    print(line.strip())

#f = open(args.output_file, 'w')
#for line in lines2:
    #f.write(line)
#f.close()
