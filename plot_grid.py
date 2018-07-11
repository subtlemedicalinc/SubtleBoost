#!/usr/bin/env python

import sys

import numpy as np
import matplotlib.pyplot as plt

import train
import subtle.subtle_plot as suplot
import subtle.subtle_preprocess as sup
import subtle.subtle_io as suio

npy_file = sys.argv[1]
idx = int(sys.argv[2])

def tile(ims):
    return np.stack(ims, axis=2)

def imshowtile(x, cmap='gray'):
    plt.imshow(x.transpose((0,2,1)).reshape((x.shape[0], -1)),cmap=cmap)

verbose = True

data = suio.load_npy_file(npy_file)

X0 = data[idx,:,:,0]
X1 = data[idx,:,:,1]
X2 = data[idx,:,:,2]

plt.figure()
plt.subplot(3,1,1)
imshowtile(tile((X0, X1, X2)))

X1mX0 = X1 - X0
#X1mX0 = np.maximum(X1mX0, 0.)

X2mX0 = X2 - X0
#X2mX0 = np.maximum(X2mX0, 0.)


plt.subplot(3,1,2)
imshowtile(tile((0*X0, X1mX0, X2mX0)))

plt.subplot(3,1,3)
imshowtile(tile((X0, X0 + X1mX0, X0 + X2mX0)))

plt.show()
