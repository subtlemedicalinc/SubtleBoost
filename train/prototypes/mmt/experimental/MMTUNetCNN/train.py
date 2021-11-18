import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.mmt import MMT as generator
from networks.discriminator import MsImageDis as discriminator
from configs.config import get_config
from trainer import trainer_brats
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
parser.add_argument('--single_decoder', action='store_true', default=False)
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--window_size', type=int,
                    default=3, help='batch_size per gpu')
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--val_freq', type=int, default=5, help='validation_frequency')
parser.add_argument('--deterministic', type=int,  default=0,
                    help='whether use deterministic training')
parser.add_argument('--base_lr_g', type=float,  default=5e-4,
                    help='network learning rate')
parser.add_argument('--base_lr_d', type=float,  default=0.001,
                    help='network learning rate')
parser.add_argument('--img_size', type=int,
                    default=192, help='input patch size of network input')
parser.add_argument('--k', type=int,
                    default=None, help='number of inputs')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=0, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='MMT', help='select one vit model')
parser.add_argument('--exp', type=str, default='MMT')
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--loss_weights', type=list, default=[5, 20, 0])
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

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

    snapshot_path = "../model/{}/{}".format(args.exp, 'MMT')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lrg' + str(args.base_lr_g) if args.base_lr_g!= 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    snapshot_path = snapshot_path + f'_opt-{args.optimizer}' if args.optimizer != 'sgd' else snapshot_path
    os.makedirs(snapshot_path, exist_ok=True)

    config = get_config(args)
    G = generator(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                out_chans = config.MODEL.SWIN.OUT_CHANS,
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
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT).cuda()

    config_d = ml_collections.ConfigDict()
    config_d.n_layer = 4
    config_d.gan_type = 'lsgan'
    config_d.dim = 64
    config_d.norm = 'bn'
    config_d.activ = 'lrelu'
    config_d.num_scales = 3
    config_d.pad_type = 'zero'
    config_d.input_dim = 1
    
    D = discriminator(config_d).cuda()
    #net.load_from(weights=np.load(config_vit.pretrained_path))

    trainer = {'BRATS': trainer_brats}
    trainer[dataset_name](args, G, D, snapshot_path)