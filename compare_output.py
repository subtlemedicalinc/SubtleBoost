#!/usr/bin/env python

import sys

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import argparse

import subtle.subtle_io as suio
import subtle.subtle_plot as suplot

usage_str = 'usage: %(prog)s [options]'
description_str = 'plot ground truth vs prediction'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--slice', action='store', dest='idx', type=int, help='show this slice (Default -- middle)', default=None)
    parser.add_argument('--truth', action='store', dest='file_truth', type=str, help='ground truth file')
    parser.add_argument('--prediction', action='store', dest='file_predict', type=str, help='prediction file')
    parser.add_argument('--output', action='store', dest='output', type=str, help='save output instead of plotting', default=None)
    parser.add_argument('--show_diff', action='store_true', dest='show_diff', help='show diff images', default=False)

    args = parser.parse_args()

    data_truth = suio.load_file(args.file_truth)
    data_predict = suio.load_file(args.file_predict)

    suplot.compare_output(data_truth, data_predict, idx=args.idx, show_diff=args.show_diff, output=args.output)

    if args.output is None:
        plt.show()
