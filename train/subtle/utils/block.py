import numpy as np
import itertools
from tqdm import tqdm

from . import io as utils_io

def load_blocks(ims, indices=None, block_size=64, strides=16):
    blocks = []
    for idxs in indices:
        ((ss, se), (xs, xe), (ys, ye)) = idxs
        block = ims[:, ss:se, xs:xe, ys:ye]

        if block.shape[1:] != [block_size] * 3:
            diff = [(0, (block_size - sh)) for sh in block.shape[1:]]
            diff = [(0, 0)] + diff
            block = np.pad(block, pad_width=diff, mode='constant', constant_values=0)
        blocks.append(block)

    return np.array(blocks)

def get_block_indices(data_file, block_size=64, strides=16, params={'h5_key': 'data'}):
    ims = utils_io.load_file(data_file, file_type='h5', params=params)
    ims_zero, _, _ = ims.transpose(1, 0, 2, 3)

    get_sweeps = lambda N: [(b*strides, (b*strides) + block_size) for b in range(((N - block_size + strides) // strides) + 1)]

    return list(itertools.product(
        get_sweeps(ims_zero.shape[0]),
        get_sweeps(ims_zero.shape[1]),
        get_sweeps(ims_zero.shape[2])
    ))

def build_block_list(data_list, block_size=64, strides=16, params={'h5_key': 'data'}):
    block_list_files = []
    block_list_indices = []
    print('Building list of 3D blocks...')
    for data_file in tqdm(data_list, total=len(data_list)):
        block_indices = get_block_indices(data_file, block_size=block_size, strides=strides, params=params)
        block_list_files.extend([data_file] * len(block_indices))
        block_list_indices.extend(block_indices)

    return np.array(block_list_files), np.array(block_list_indices)

def is_valid_block(block, block_size=64, pixel_percent=0.1):
    filt_blocks = []
    get_nz = lambda plist: np.sum([len(np.nonzero(sl)[0]) for sl in plist])
    sel_idx = []

    percent = get_nz(block.reshape((block_size, block_size**2))) / (block_size ** 3)
    return (percent >= pixel_percent)
