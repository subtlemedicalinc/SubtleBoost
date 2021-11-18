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
from tqdm import tqdm
from torchvision import transforms
from datasets.dataset_brats import BRATS_dataset, RandomGenerator
from torchvision.utils import make_grid
from evaluator import evaluator_brats, AverageMeter

input_combination = [[0], [1], [2], [3], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [0, 1, 2],
                     [0, 1, 3], [0, 2, 3], [1, 2, 3]]

val_combination = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2] ]


def random_missing_input(n, k=None):
    contrasts = list(range(n))
    random.shuffle(contrasts)
    if not k:
        k = np.random.randint(1, n)   #k: number of inputs in [1, n_contrast)
    contrast_input = sorted(contrasts[:k])
    contrast_output = sorted(contrasts[k:])
    return contrast_input, contrast_output


def make_image_grid(tensor_list):
    images = []
    for tensor in tensor_list:
        image = tensor[0, 0, :, :].unsqueeze(0)
        image = (image - image.min()) / (image.max() - image.min())
        image = image.repeat(3,1,1)
        images.append(image)
    image_grid = make_grid(images)
    return image_grid


def random_split_data(data,  k=None):
    n_contrast = len(data)
    contrasts = list(range(n_contrast))
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
    return loss


def trainer_brats(args, G, D, snapshot_path):

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr_g = args.base_lr_g
    base_lr_d = args.base_lr_d
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = BRATS_dataset(base_dir=args.root_path, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(flip=True, scale=[0.9, 1.1])]))
    db_val = BRATS_dataset(base_dir=args.root_path, split="val",
                               transform=transforms.Compose(
                                   [RandomGenerator(flip=False, scale=None)]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)
    if args.optimizer == 'sgd':
        opt_G = optim.SGD(G.parameters(), lr=base_lr_g, momentum=0.9, weight_decay=0.0001)
        opt_D = optim.SGD(D.parameters(), lr=base_lr_d, momentum=0.9, weight_decay=0.0001)
    else:
        opt_G = optim.Adam(G.parameters(), lr=base_lr_g, weight_decay=0.00001)
        opt_D = optim.Adam(D.parameters(), lr=base_lr_d, weight_decay=0.00001)

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance_mae = np.inf
    best_performance_mse = np.inf
    best_performance_psnr = np.inf
    best_performance_ssim = np.inf
    iterator = tqdm(range(args.start_epoch, max_epoch))
    criterion_img = L1Loss()
    criterion_latent = L1Loss()
    model_G = G.module if args.n_gpu>1 else G
    model_D = D.module if args.n_gpu>1 else D

    loss_weights = args.loss_weights
    for epoch_num in iterator:
        G.train()
        D.train()
        loss_meter_g = AverageMeter()
        loss_meter_gan_g = AverageMeter()
        loss_meter_gan_d = AverageMeter()
        loss_meter_rec = AverageMeter()
        n_contrast = model_G.n_contrast
        contrasts = list(range(n_contrast))
        for i_batch, data in enumerate(tqdm(trainloader)):
            data = [d.detach().cuda() for d in data]
            img_inputs, img_targets, contrast_inputs, contrast_outputs = random_split_data(data, args.k)
            img_outs, _ = G(img_inputs, contrast_inputs, contrasts)

            img_inputs_recon = [img_outs[contrast_i] for contrast_i in contrast_inputs]
            img_outputs = [img_outs[contrast_o] for contrast_o in contrast_outputs]

            loss_input_rec = recon_loss(img_inputs_recon, img_inputs, criterion_img)
            loss_output_rec = recon_loss(img_outputs, img_targets, criterion_img)

            real_labels = [contrast + 1 for contrast in contrast_outputs]
            loss_gan_d = model_D.calc_dis_loss([img_output.detach() for img_output in img_outputs], img_targets, real_labels)
            loss_gan_g = model_D.calc_gen_loss(img_outputs, real_labels)

            opt_G.zero_grad()
            loss_g = loss_weights[0]*loss_input_rec + loss_weights[1]*loss_output_rec + loss_weights[2]*loss_gan_g
            loss_g.backward()
            opt_G.step()

            opt_D.zero_grad()
            loss_d = loss_weights[2]*loss_gan_d
            loss_d.backward()
            opt_D.step()

            loss_meter_gan_d.update(loss_gan_d.item())
            loss_meter_gan_g.update(loss_gan_g.item())
            loss_meter_g.update(loss_g.item())
            loss_meter_rec.update(loss_output_rec.item())

            iter_num = iter_num + 1
            logging.info(f'iteration {iter_num} : loss_g: {loss_g.item()}, loss_d: {loss_d.item()}')
            writer.add_scalar('train/loss_g', loss_g, iter_num)
            writer.add_scalar('train/loss_d', loss_d, iter_num)
            writer.add_scalar('train/loss_gan_d', loss_gan_d, iter_num)
            writer.add_scalar('train/loss_gan_g', loss_gan_g, iter_num)
            writer.add_scalar('train/loss_input_rec', loss_input_rec, iter_num)
            writer.add_scalar('train/loss_output_rec', loss_output_rec, iter_num)

            if iter_num % 20 == 0:
                writer.add_image('train/inputs', make_image_grid(img_inputs), iter_num)
                writer.add_image('train/targets', make_image_grid(img_targets), iter_num)
                writer.add_image('train/outputs', make_image_grid(img_outputs), iter_num)
                writer.add_image('train/inputs_recon', make_image_grid(img_inputs_recon), iter_num)

        writer.add_scalar('train/epoch_loss_g', loss_meter_g.avg, epoch_num)
        writer.add_scalar('train/epoch_loss_gan_g', loss_meter_gan_g.avg, epoch_num)
        writer.add_scalar('train/epoch_loss_gan_d', loss_meter_gan_d.avg, epoch_num)
        writer.add_scalar('train/epoch_loss_output_rec', loss_meter_rec.avg, epoch_num)

        # save ckpt every epoch
        save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        torch.save({'G':model_G.state_dict(), 'D':model_D.state_dict(), 'epoch': epoch_num}, save_mode_path)
        logging.info("save model to {}".format(save_mode_path))

        val_metrics = evaluator_brats(args, model_G, val_combination, split='val')
        performance_mae = 0
        performance_mse = 0
        performance_ssim = 0
        performance_psnr = 0
        for i in range(len(val_combination)):
            writer.add_scalar(f'val/ssim_{i}', val_metrics[i][0]['ssim'], epoch_num)
            performance_ssim += val_metrics[i][0]['ssim']
            writer.add_scalar(f'val/mae_{i}', val_metrics[i][0]['mae'], epoch_num)
            performance_mae += val_metrics[i][0]['mae']
            writer.add_scalar(f'val/psnr_{i}', val_metrics[i][0]['psnr'], epoch_num)
            performance_psnr += val_metrics[i][0]['psnr']
            writer.add_scalar(f'val/mse_{i}', val_metrics[i][0]['mse'], epoch_num)
            performance_mse += val_metrics[i][0]['mse']
        if performance_mse < best_performance_mse:
            best_performance_mse = performance_mse
            save_mode_path = os.path.join(snapshot_path, 'best_mse.pth')
            torch.save({'G':model_G.state_dict(), 'D':model_D.state_dict(), 'epoch': epoch_num}, save_mode_path)
        if performance_mae < best_performance_mae:
            best_performance_mae = performance_mae
            save_mode_path = os.path.join(snapshot_path, 'best_mae.pth')
            torch.save({'G':model_G.state_dict(), 'D':model_D.state_dict(), 'epoch': epoch_num}, save_mode_path)
        if performance_ssim < best_performance_ssim:
            best_performance_ssim = performance_ssim
            save_mode_path = os.path.join(snapshot_path, 'best_ssim.pth')
            torch.save({'G':model_G.state_dict(), 'D':model_D.state_dict(), 'epoch': epoch_num}, save_mode_path)
        if performance_psnr < best_performance_psnr:
            best_performance_psnr = performance_psnr
            save_mode_path = os.path.join(snapshot_path, 'best_psnr.pth')
            torch.save({'G':model_G.state_dict(), 'D':model_D.state_dict(), 'epoch': epoch_num}, save_mode_path)
        if epoch_num >= max_epoch - 1:
            iterator.close()
            break

    writer.close()
    return "Training Finished!"