#!/usr/bin/env python

import sys
import numpy as np

z = np.load(sys.argv[1], mmap_mode='r')
print(sys.argv[1], z.shape)
