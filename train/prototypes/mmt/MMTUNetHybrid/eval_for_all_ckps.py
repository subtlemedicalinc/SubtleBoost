import argparse
import os
from glob import glob
import numpy as np
import torch
from evaluator import evaluator_brats, evaluator_ixi
from networks.mmt import MMT as generator
from torchvision.utils import save_image
from tqdm import tqdm
from configs.config import get_config
from datasets.dataset_brats import BRATS_dataset, RandomGeneratorBRATS
from datasets.dataset_ixi import IXI_dataset, RandomGeneratorIXI
from torchvision import transforms
from torch.nn import Upsample
import json
import pandas as pd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

split = 'val'
data_dir = '/mnt/datasets/srivathsa/jiang_raid/projects/SubtleGAN/data/brats2021_slices_crop192x160'
ckpt_base = '/home/srivathsa/projects/SubtleGad/train/prototypes/mmt/MMTUNetHybrid/model/'
ckpt_base += 'MMT_single_no_gan_epo100_bs24_lrg0.0005_5.0_20.0_0.0_0.0_vgg0'

model_type = 'single' # ('random', 'zerogad', 'single', 'mra_synth')
dataset = 'brats'
cca = True #cross-contrast-attn

if model_type == 'mra_synth' or dataset == 'ixi':
    db = IXI_dataset(
        base_dir=data_dir, split=split,
        transform=transforms.Compose([
            RandomGeneratorIXI(flip=False, scale=None, n_contrast=3)
        ])
    )
    img_size=[256, 256]
    window_size=[8, 8]
    batch_size = 6
    evaluator = evaluator_ixi
else:
    db = BRATS_dataset(
        base_dir=data_dir, split=split,
        transform=transforms.Compose([
            RandomGeneratorBRATS(flip=False, scale=None)
        ])
    )
    if 'full_brain_256' in data_dir:
        img_size=[256, 256]
        window_size=[8, 8]
        batch_size = 12
    else:
        img_size=[160, 192]
        window_size=[5, 6]
        batch_size = 24
    evaluator = evaluator_brats

G = generator(img_size=img_size,
                                patch_size=4,
                                in_chans=1,
                                out_chans = 1,
                                embed_dim=96,
                                depths=[2,2,2,2],
                                num_heads=[3,6,12,24],
                                window_size=window_size,
                                mlp_ratio=4.0,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0,
                                drop_path_rate=0,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False, cross_contrast_attn=cca).cuda()

args = argparse.Namespace()
args.batch_size = batch_size
args.n_gpu = 1
args.root_path = data_dir
args.seed = 1234
args.mra_synth = (model_type == 'mra_synth')

metric_list = []
epoch_list = []
met_path = '{}/metrics_list.json'.format(ckpt_base)

if os.path.exists(met_path):
    metric_list = pd.read_json(met_path).to_dict(orient='records')
    epoch_list = [r['epoch'] for r in metric_list]

ckp_files = [
    f for f in glob('{}/epoch*.pth'.format(ckpt_base))
    if f.split('/')[-1] not in epoch_list
]
ckp_files = sorted(ckp_files, key=lambda f: int(f.split('/')[-1].split('.')[0].split('_')[1]))

input_comb = {
    'mra_synth': [
        [0, 1]
    ],
    'ixi_single': [
        [0, 1], [0, 2], [1, 2]
    ],
    'zerogad': [
        [0, 2, 3]
    ],
    'single': [
        [0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3]
    ],
    'random': [
        [0], [1], [2], [3],
        [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],
        [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]
    ]
}

for ckpt_path in ckp_files:
    print('**** EVALUATING {} ****'.format(ckpt_path))

    state_dict = torch.load(ckpt_path, map_location='cuda:0')
    G.load_state_dict(state_dict['G'])
    G.eval()

    eval_args = {
        'args': args,
        'model': G,
        'input_combination': input_comb[model_type],
        'split':split
    }

    if model_type == 'mra_synth':
        eval_args['mra_synth'] = True

    metrics, _, _ = evaluator(**eval_args)
    met_dicts = []
    for met_list in metrics:
        mdict = met_list[0]
        mdict['epoch'] = ckpt_path.split('/')[-1]
        met_dicts.append(mdict)

    metric_list.extend(met_dicts)

with open('{}/metrics_list.json'.format(ckpt_base), 'w') as f:
    f.write(json.dumps(metric_list))

print('Saved metrics list to {}'.format(ckpt_base))
