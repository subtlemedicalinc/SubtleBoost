#!/usr/bin/env python

import sys

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import argparse

import subtle.utils.io as utils_io
import subtle.subtle_plot as suplot

usage_str = 'usage: %(prog)s [options]'
description_str = 'plot ground truth vs prediction'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--all_slices', action='store_true', dest='all_slices',  help='show all slices', default=False)
    parser.add_argument('--slice', action='append', dest='idxs', type=int, help='show this slice (Default -- middle)', default=[])
    parser.add_argument('--truth', action='store', dest='file_truth', type=str, help='ground truth file')
    parser.add_argument('--prediction', action='store', dest='file_predict', type=str, help='prediction file')
    parser.add_argument('--output', action='store', dest='output', type=str, help='save output', default=None)
    parser.add_argument('--plot', action='store_true', dest='plot', help='plot output', default=False)
    parser.add_argument('--show_diff', action='store_true', dest='show_diff', help='show diff images', default=False)

    args = parser.parse_args()

    data_truth = utils_io.load_file(args.file_truth)
    data_predict = utils_io.load_file(args.file_predict)

    n_slices = data_truth.shape[0]

    print("Number of slices:", n_slices)
    print()

    if args.all_slices:
        args.idxs = range(n_slices)

    if len(args.idxs) == 0:
        suplot.compare_output(data_truth, data_predict, idx=None, show_diff=args.show_diff, output=args.output)
    else:
        for idx in sorted(args.idxs):
            if idx < n_slices and idx >= 0:
                suplot.compare_output(data_truth, data_predict, idx=idx, show_diff=args.show_diff, output=args.output)

    if args.plot:
        plt.show()
