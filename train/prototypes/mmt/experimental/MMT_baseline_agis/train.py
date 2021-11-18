import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.baseline_agis import MMT_baseline_agis as generator
from networks.discriminator import McImageDis as discriminator
from configs.config import get_config
from trainer_brats import trainer_brats
from trainer_ixi import trainer_ixi
import ml_collections




parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/jiang/projects/SubtleGAN/data/brats2021_slices', help='root dir for data')
parser.add_argument('--cfg', type=str, default='configs/swin_tiny_patch4_window6_192_medium.yml')
parser.add_argument('--dataset', type=str,
                    default='BRATS', help='experiment_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=50, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--val_freq', type=int, default=1, help='validation_frequency')
parser.add_argument('--deterministic', type=int,  default=0,
                    help='whether use deterministic training')
parser.add_argument('--base_lr_g', type=float,  default=5e-4,
                    help='network learning rate')
parser.add_argument('--img_size', type=int,
                    default=192, help='input patch size of network input')
parser.add_argument('--n_contrast', type=int, default=4, help='total number of contrast in the dataset')
parser.add_argument('--k', type=int,
                    default=None, help='number of inputs')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=0, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='MMT', help='select one vit model')
parser.add_argument('--exp', type=str, default='MMT', help='name of experiment')
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--margin', type=float, default=0.1)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--warmup_epoch', type=int, default=3)
parser.add_argument('--vis_freq', type=int, default=50)
parser.add_argument('--zero_gad', action="store_true", default=False)


args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    args.is_pretrain = False

    snapshot_path = "model/{}/{}".format('MMT', args.exp)
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size*args.n_gpu)
    snapshot_path = snapshot_path + '_lrg' + str(args.base_lr_g) if args.base_lr_g!= 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    snapshot_path = snapshot_path + f'_opt-{args.optimizer}' if args.optimizer != 'sgd' else snapshot_path
    os.makedirs(snapshot_path, exist_ok=True)


    G = generator(n_contrast=args.n_contrast).cuda()
    trainer = {'BRATS': trainer_brats, 'IXI': trainer_ixi}
    trainer[dataset_name](args, G, snapshot_path)