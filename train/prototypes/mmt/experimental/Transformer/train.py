import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.generator import SubtleGeneratorUNet as generator
from networks.generator import CONFIGS as CONFIGS_ViT_seg
from networks.discriminator import MsImageDis as discriminator
from trainer import trainer_brats

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/jiang/projects/SubtleGAN/data/brats2021_slices', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='BRATS', help='experiment_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--single_decoder', action='store_true', default=False)
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--window_size', type=int,
                    default=3, help='batch_size per gpu')
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--enc_layers', type=int,
                    default=3, help='number of encoder blocks')
parser.add_argument('--dec_layers', type=int,
                    default=3, help='number of decoder blocks')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--val_freq', type=int, default=5, help='validation_frequency')
parser.add_argument('--deterministic', type=int,  default=0,
                    help='whether use deterministic training')
parser.add_argument('--base_lr_g', type=float,  default=0.001,
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
                    default='ViT-L_8', help='select one vit model')
parser.add_argument('--exp', type=str, default='MMT')
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--loss_weights', type=list, default=[10, 20, 0.001])
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
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


    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.window_size = args.window_size

    snapshot_path = "../model/{}/{}".format(args.exp, 'MMT')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lrg' + str(args.base_lr_g) if args.base_lr_g!= 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path
    snapshot_path = snapshot_path + '_single-decoder' if args.single_decoder else snapshot_path
    snapshot_path = snapshot_path + f'_enc-layers{args.enc_layers}' if args.enc_layers != 3 else snapshot_path
    snapshot_path = snapshot_path + f'_dec-layers{args.dec_layers}' if args.dec_layers != 3 else snapshot_path
    snapshot_path = snapshot_path + f'_window-size-{config_vit.window_size}' if config_vit.window_size != 3 else snapshot_path
    snapshot_path = snapshot_path + f'_opt-{args.optimizer}' if args.optimizer != 'sgd' else snapshot_path
    os.makedirs(snapshot_path, exist_ok=True)

    config_vit.single_decoder = args.single_decoder
    config_vit.transformer.encoder_num_layers = args.enc_layers
    config_vit.transformer.decoder_num_layers = args.dec_layers
    G = generator(config_vit, img_size=args.img_size).cuda()
    D = discriminator(config_vit).cuda()
    #net.load_from(weights=np.load(config_vit.pretrained_path))

    trainer = {'BRATS': trainer_brats}
    trainer[dataset_name](args, G, D, snapshot_path)