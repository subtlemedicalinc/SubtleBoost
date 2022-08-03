import argparse
import os
import glob
import numpy as np
import torch
import random
from networks.mmt import MMT as generator
from torchvision.utils import save_image
from tqdm import tqdm
from configs.config import get_config
from datasets.dataset_brats import BRATS_dataset, RandomGeneratorBRATS
from datasets.dataset_ixi import IXI_dataset, RandomGeneratorIXI
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.functional import interpolate
from torch.nn.functional import normalize


def split_data(data, inputs, targets):
    contrast_input = inputs
    contrast_output = targets
    img_inputs = [data[i].detach().cuda() for i in contrast_input]
    img_targets = [data[i].detach().cuda() for i in contrast_output]
    return img_inputs, img_targets, contrast_input, contrast_output


def sum_attn(attn_maps, head_normalization=True):
    n_heads = [24, 24, 12, 12, 6, 6, 3, 3]
    attn_scores = []
    for n_head, attn_map in zip(n_heads, attn_maps):
        attn_score = torch.sum(attn_map, dim=[0, 2, 3, 4])
        if head_normalization:
            attn_score = attn_score/n_head
        attn_scores.append(attn_score)
    attn_scores = torch.stack(attn_scores, dim=0)
    return attn_scores


input_combination_all = [[0], [1], [2], [3], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [0, 1, 2],
                     [0, 1, 3], [0, 2, 3], [1, 2, 3]]

input_combination_3 = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]

input_combination_zerogad = [[0, 2, 3]]
input_combination_2 = [[1, 2], [0, 2], [0, 1]]

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/mnt/raid/jiang/projects/SubtleGAN/data/brats2021_slices_crop192x160', help='root dir for data')
parser.add_argument('--cfg', type=str, default='configs/mmt.yml')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--dataset', type=str,
                    default='BRATS', help='experiment_name')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--k', type=int,
                    default=None, help='number of inputs')
parser.add_argument('--zero_gad', action='store_true', default=False, help='eval zero_gad')
parser.add_argument('--head_norm', action='store_true', default=False, help='normalize attn maps by # of heads')
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--n_contrast', type=int, default=4, help='total number of contrast in the dataset')


args = parser.parse_args()

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    args.n_contrast = 4 if args.dataset == 'BRATS' else 3
    args.cfg = 'configs/mmt.yml' if args.dataset == 'BRATS' else 'configs/mmt_ixi.yml'
    config = get_config(args)
    
    G = generator(img_size=config.DATA.IMG_SIZE,
                  patch_size=config.MODEL.SWIN.PATCH_SIZE,
                  in_chans=config.MODEL.SWIN.IN_CHANS,
                  out_chans=config.MODEL.SWIN.OUT_CHANS,
                  embed_dim=config.MODEL.SWIN.EMBED_DIM,
                  depths=config.MODEL.SWIN.DEPTHS,
                  num_heads=config.MODEL.SWIN.NUM_HEADS,
                  window_size=config.MODEL.SWIN.WINDOW_SIZE,
                  mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                  qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                  qk_scale=config.MODEL.SWIN.QK_SCALE,
                  drop_rate=config.MODEL.DROP_RATE,
                  drop_path_rate=config.MODEL.DROP_PATH_RATE,
                  ape=config.MODEL.SWIN.APE,
                  patch_norm=config.MODEL.SWIN.PATCH_NORM,
                  use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                  seg=False,
                  num_contrast=args.n_contrast).cuda()

    state_dict = torch.load(os.path.join(args.model_path, args.ckpt), map_location='cpu')
    G.load_state_dict(state_dict['G'])
    if args.n_gpu > 1:
        G = nn.DataParallel(G)
    G = G.cuda()
    G.eval()

    if args.zero_gad:
        input_combination = input_combination_zerogad
    elif args.k == 3:
        input_combination = input_combination_3
    elif args.k == 2:
        input_combination = input_combination_2
    else:
        input_combination = input_combination_all

    batch_size = args.batch_size * args.n_gpu
    if args.dataset == 'BRATS':
        db = BRATS_dataset(base_dir=args.root_path, split=args.split,
                           transform=transforms.Compose(
                               [RandomGeneratorBRATS(flip=False, scale=None)]))
    else:
        db = IXI_dataset(base_dir=args.root_path, split=args.split,
                           transform=transforms.Compose(
                               [RandomGeneratorIXI(flip=False, scale=None)]))
    print("The length of data set is: {}".format(len(db)))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    dataloader = DataLoader(db, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                            worker_init_fn=worker_init_fn)
    with open(os.path.join(args.model_path, f"attn_scores_{args.split}_hn_{args.head_norm}.txt"), "w") as f:
        for inputs in input_combination:
            targets = list(set(range(args.n_contrast)) - set(inputs))
            scores = torch.zeros(8, len(inputs)).cuda()
            with torch.no_grad():
                for i_batch, data in enumerate(tqdm(dataloader)):
                    data = [d.detach().cuda() for d in data]
                    img_inputs, img_targets, contrast_input, contrast_output = split_data(data, inputs, targets)
                    _, _, attn_maps = G(img_inputs, contrast_input, contrast_output, return_attention=True)
                    attn_scores = sum_attn(attn_maps, head_normalization=args.head_norm)
                    scores += attn_scores
            normalized_scores = normalize(scores, p=1)
            overall_scores = normalize(torch.sum(scores, dim=0), p=1, dim=0)
            msg = f'Inputs {inputs}: \n {normalized_scores} \n Overall: {overall_scores}'
            f.write(msg)
            print(msg)
        





