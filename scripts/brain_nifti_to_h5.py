#!/usr/bin/env python

import nibabel as nib
import subtle.subtle_io as suio
import numpy as np

import os
import sys

mydir = sys.argv[1]


# try to find the nifti files with the names, "BrainExtractionBrain.nii.gz"

nifti_files = []
for dir_name, subdir_list, file_list in os.walk(mydir):
    for file_name in file_list:
        if "BrainExtractionBrain" in file_name:
            nifti_files.append(os.path.join(dir_name, file_name))

# now that we got them, sort them into the right order
nifti_files_numbers = [int(os.path.basename(f).split('_')[1]) for f in nifti_files]

idx_sort = np.argsort(nifti_files_numbers)
nifti_files_sorted = [nifti_files[i] for i in idx_sort]

x0 = nib.load(nifti_files_sorted[0]).get_fdata()
x1 = nib.load(nifti_files_sorted[1]).get_fdata()
x2 = nib.load(nifti_files_sorted[2]).get_fdata()

print(x0.shape, x1.shape, x2.shape)

xx = np.stack((x0, x1, x2), axis=1)

suio.save_data('{}/{}_brain.h5'.format(os.path.dirname(mydir), os.path.basename(mydir)), xx)
