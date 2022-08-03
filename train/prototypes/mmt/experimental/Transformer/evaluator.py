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
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def split_data(data, inputs, targets):
    contrast_input = inputs
    contrast_output = targets
    img_inputs = [data[i].detach().cuda() for i in contrast_input]
    img_targets = [data[i].detach().cuda() for i in contrast_output]
    return img_inputs, img_targets, contrast_input, contrast_output


def recon_loss(outputs, targets, criterion):
    loss = 0
    for i in range(len(outputs)):
        loss += criterion(outputs[i], targets[i])
    return loss


def eval_brats(model, inputs, targets, dataloader):
    metrics_meters = [[AverageMeter() for _ in range(4)] for _ in range(len(targets))]
    model.eval()
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(dataloader)):
            data = [d.detach().cuda() for d in data]
            img_inputs, img_targets, contrast_input, contrast_output = split_data(data, inputs, targets)
            outputs  = model(img_inputs, contrast_input, contrast_output)
            img_outputs = outputs[0]
            for i, outputs in enumerate(img_outputs):
                output_imgs = outputs.detach().cpu().numpy()
                target_imgs = img_targets[i].detach().cpu().numpy()
                for j in range(output_imgs.shape[0]):
                    img_o = output_imgs[j, 0, :, :]
                    img_t = target_imgs[j, 0, :, :]
                    img_max = img_t.max()
                    img_o /= img_max
                    img_t /= img_max
                    img_o = img_o.clip(0, 1)
                    metrics_meters[i][0].update(mse(img_o, img_t))
                    metrics_meters[i][1].update(mae(img_o, img_t))
                    metrics_meters[i][2].update(ssim(img_o, img_t))
                    metrics_meters[i][3].update(psnr(img_t, img_o))
    metrics = [{} for _ in range(len(targets))]
    for i in range(len(targets)):
        metrics[i]['mse'] = metrics_meters[i][0].avg
        metrics[i]['mae'] = metrics_meters[i][1].avg
        metrics[i]['ssim'] = metrics_meters[i][2].avg
        metrics[i]['psnr'] = metrics_meters[i][3].avg
    print(f"***Inputs: {inputs}; Outputs: {targets}; {metrics}")
    return metrics


def evaluator_brats(args, model, input_combination, split='test'):

    batch_size = args.batch_size * args.n_gpu
    db = BRATS_dataset(base_dir=args.root_path, split=split,
                               transform=transforms.Compose(
                                   [RandomGenerator(flip=False, scale=None)]))

    print("The length of data set is: {}".format(len(db)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    dataloader = DataLoader(db, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    metrics = []
    for inputs in input_combination:
        targets = list(set(range(4)) - set(inputs))
        metric = eval_brats(model, inputs, targets, dataloader)
        metrics.append(metric)
    return metrics