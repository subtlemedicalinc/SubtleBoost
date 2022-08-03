import argparse
import logging
import os
import pdb
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import L1Loss, MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from torchvision import transforms
from datasets.dataset_brats import BRATS_dataset, RandomGenerator
from evaluator_brats import evaluator_brats, AverageMeter
import pdb
from timm.scheduler.cosine_lr import CosineLRScheduler
from utils import list2str, make_image_grid


input_combination = [[0], [1], [2], [3], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [0, 1, 2],
                     [0, 1, 3], [0, 2, 3], [1, 2, 3]]

input_combination_3 = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]



def cosine_similarity(x, y):
    cos = nn.CosineSimilarity(dim=0)
    return cos(x.view(-1), y.view(-1))


# compute triplet loss for each sample
def compute_triplet_loss(sample_1, sample_2, margin=0.1):
    triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=cosine_similarity, margin=margin)
    n_contrast = sample_1.shape[0]
    loss = 0
    for i in range(n_contrast):
        for j in range(n_contrast):
                anchor1 = sample_1[i]
                positive1 = sample_1[j]
                negative1 = sample_2[j]
                anchor2 = sample_2[i]
                positive2 = sample_2[j]
                negative2 = sample_1[j]
                loss += triplet_loss(anchor1, positive1, negative1) + triplet_loss(anchor2, positive2, negative2)
    loss = loss/(2*n_contrast**2)
    return loss


def contrastive_loss(enc_outs, idx_i, idx_j, margin=0.1):
    loss = 0
    for enc_out in enc_outs:
        # enc_out: B n_contrast H W C
        sample_i = enc_out[idx_i, :, :, :, :]
        sample_j = enc_out[idx_j, :, :, :, :]
        loss += compute_triplet_loss(sample_i, sample_j, margin)
    loss = loss/len(enc_outs)
    return loss


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def random_missing_input(n, k=None):
    contrasts = list(range(n))
    random.shuffle(contrasts)
    if not k:
        k = np.random.randint(1, n)   #k: number of inputs in [1, n_contrast)
    contrast_input = sorted(contrasts[:k])
    contrast_output = sorted(contrasts[k:])
    return contrast_input, contrast_output


def random_split_data(data,  k=None, zero_gad=False):
    n_contrast = len(data)
    contrasts = list(range(n_contrast))
    if zero_gad:
        contrast_input = [0, 2, 3]
        contrast_output = [1]
    else:
        random.shuffle(contrasts)
        if not k:
            k = np.random.randint(1, n_contrast) #k: number of inputs in [1, n_contrast)
        contrast_input = sorted(contrasts[:k])
        contrast_output = sorted(contrasts[k:])
    img_inputs = [data[i]  for i in contrast_input]
    img_targets = [data[i] for i in contrast_output]
    return img_inputs, img_targets, contrast_input, contrast_output


def recon_loss(outputs, targets, criterion):
    loss = 0
    for output, target in zip(outputs, targets):
        loss += criterion(output, target)
    loss = loss/len(outputs)
    return loss


def trainer_brats(args, G, snapshot_path):
    best_performance_mae = np.inf
    best_performance_mse = np.inf
    best_performance_psnr = 0
    best_performance_ssim = 0

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr_g = args.base_lr_g
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = BRATS_dataset(base_dir=args.root_path, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(flip=True, scale=[0.9, 1.1])]))

    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        G = nn.DataParallel(G)

    opt_G = optim.Adam(G.parameters(), lr=base_lr_g)


    if args.ckpt:
        state_dict = torch.load(args.ckpt, map_location='cpu')
        G.load_state_dict(state_dict['G'])
        args.start_epoch = state_dict['epoch']
        best_performance_mae = state_dict['mae']
        best_performance_mse = state_dict['mse']
        best_performance_psnr = state_dict['psnr']
        best_performance_ssim = state_dict['ssim']
        opt_G.load_state_dict(state_dict['opt_G'])

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    criterion_img = L1Loss()
    model_G = G.module if args.n_gpu>1 else G

    for epoch_num in trange(args.start_epoch, max_epoch):
        G.train()
        loss_meter_g = AverageMeter()
        loss_meter_rec = AverageMeter()
        for i_batch, data in enumerate(tqdm(trainloader)):
            data = [d.detach().cuda() for d in data]
            iter_num = iter_num + 1
            img_inputs, img_targets, contrast_inputs, contrast_outputs = random_split_data(data, args.k, zero_gad=args.zero_gad)

            img_codes = G.encode_imgs(img_inputs, contrast_inputs)
            z = torch.stack(img_codes, dim=1)

            ## compute cost c1
            loss_g1 = 0
            for img_code in img_codes:
                img_outputs = G.decode_imgs(img_code, contrast_outputs)
                loss_g1 += recon_loss(img_outputs, img_targets, criterion_img)
            loss_g1 /= len(img_inputs)

            ## compute cost c2
            z_var = torch.var(z, dim=1, unbiased=False)
            loss_g2 = torch.mean(z_var)


            ## compute cost c3
            # fuse latent codes
            z_fused, _ = torch.max(z, dim=1)
            img_outputs = G.decode_imgs(z_fused, contrast_outputs)
            loss_g3 = recon_loss(img_outputs, img_targets, criterion_img)

            loss_g = loss_g1 + loss_g2 + loss_g3
            writer.add_scalar('train/loss_g1', loss_g1, iter_num)
            writer.add_scalar('train/loss_g2', loss_g2, iter_num)
            writer.add_scalar('train/loss_g3', loss_g3, iter_num)
            writer.add_scalar('train/loss_g', loss_g, iter_num)


            # update generator
            opt_G.zero_grad()
            loss_g.backward()
            opt_G.step()
            loss_meter_g.update(loss_g.item())
            loss_meter_rec.update(loss_g3.item())

            if iter_num % args.vis_freq == 0:
                writer.add_image('train/inputs', make_image_grid(img_inputs), iter_num)
                writer.add_image('train/targets', make_image_grid(img_targets), iter_num)
                writer.add_image('train/outputs', make_image_grid(img_outputs), iter_num)

        # log epoch loss
        writer.add_scalar('epoch/loss_g', loss_meter_g.avg, epoch_num)
        writer.add_scalar('epoch/loss_g3', loss_meter_rec.avg, epoch_num)

        state_dict = {'G': model_G.state_dict(), 'epoch': epoch_num, 'opt_G': opt_G.state_dict()}

        # test on validation set
        if epoch_num % args.val_freq == 0:
            if args.zero_gad:
                val_combination = [[0, 2, 3]]
            elif args.k == 3:
                val_combination = input_combination_3
            else:
                val_combination = input_combination
            val_metrics = evaluator_brats(args, G, val_combination, split='val')
            performance_mae = 0
            performance_mse = 0
            performance_ssim = 0
            performance_psnr = 0
            for i in range(len(val_combination)):
                output_combination = list(set(range(4)) - set(val_combination[i]))
                for j in range(len(val_metrics[i])):
                    writer.add_scalar(f'val_{list2str(val_combination[i])}/ssim_{output_combination[j]}', val_metrics[i][j]['ssim'], epoch_num)
                    performance_ssim += val_metrics[i][j]['ssim']
                    writer.add_scalar(f'val_{list2str(val_combination[i])}/mae_{output_combination[j]}', val_metrics[i][j]['mae'], epoch_num)
                    performance_mae += val_metrics[i][j]['mae']
                    writer.add_scalar(f'val_{list2str(val_combination[i])}/psnr_{output_combination[j]}', val_metrics[i][j]['psnr'], epoch_num)
                    performance_psnr += val_metrics[i][j]['psnr']
                    writer.add_scalar(f'val_{list2str(val_combination[i])}/mse_{output_combination[j]}', val_metrics[i][j]['mse'], epoch_num)
                    performance_mse += val_metrics[i][j]['mse']
            if performance_mse < best_performance_mse:
                best_performance_mse = performance_mse
                save_path = os.path.join(snapshot_path, 'best_mse.pth')
                state_dict['mse'] = best_performance_mse
                torch.save(state_dict, save_path)
            if performance_mae < best_performance_mae:
                best_performance_mae = performance_mae
                save_path = os.path.join(snapshot_path, 'best_mae.pth')
                state_dict['mae'] = best_performance_mae
                torch.save(state_dict, save_path)
            if performance_ssim >= best_performance_ssim:
                best_performance_ssim = performance_ssim
                save_path = os.path.join(snapshot_path, 'best_ssim.pth')
                state_dict['ssim'] = best_performance_ssim
                torch.save(state_dict, save_path)
            if performance_psnr >= best_performance_psnr:
                best_performance_psnr = performance_psnr
                save_path = os.path.join(snapshot_path, 'best_psnr.pth')
                state_dict['psnr'] = best_performance_psnr
                torch.save(state_dict, save_path)

        # save ckpt every epoch
        state_dict['ssim'] = best_performance_ssim
        state_dict['mse'] = best_performance_mse
        state_dict['mae'] = best_performance_mae
        state_dict['psnr'] = best_performance_psnr
        save_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        torch.save(state_dict, save_path)
        logging.info("save model to {}".format(save_path))

    writer.close()
    return "Training Finished!"