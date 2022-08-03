from modules.helpers import create_dataloaders, Resize, ToTensor
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from torchvision.utils import make_grid, save_image
from kornia.geometry.transform import resize
from tqdm import tqdm
import itertools
import torch
import os
import torch
import argparse
import pdb
import cv2
import glob
import numpy as np

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

def make_image_grid(image_array):
    images = []
    n = image_array.shape[0]
    for i in range(n):
        image = cv2.resize(image_array[i], (160, 192))
        image = torch.from_numpy(image).unsqueeze(0)
        image = (image - image.min()) / (image.max() - image.min())
        image = image.repeat(3,1,1)
        image = torch.rot90(image, 2, [1, 2])
        images.append(image)
    image_grid = make_grid(images, padding=0)
    return image_grid

def list2str(x):
    f = ''
    for x_i in x:
        f += f'{x_i}'
    return f

# def list2str(x):
#     map_list = [0, 2, 1, 3]
#     x_mapped = [x[map_list[i]] for i in range(len(map_list))]
#     f = ''
#     for i, x_i in enumerate(x_mapped):
#         if x_i == 1:
#             f += f'{i}'
#     return f


def eval_brats(generator, scenario, patient_list, model_path, vis_dir='vis'):
    scenario_tag = list2str(scenario)
    scenario = np.array(scenario)
    generator.eval()
    impute_tensor = torch.zeros((1, 256, 256), device='cuda')
    metrics_meters = [[AverageMeter() for _ in range(4)] for _ in range(np.sum(scenario==0))]
    # pdb.set_trace()
    with torch.no_grad():
        for patient in tqdm(patient_list):
            pat_name = patient.split("/")[-1]
            print(f'Scenario {scenario_tag}: evaluating {pat_name}')
            if vis_dir is not None:
                save_dir = os.path.join(model_path, vis_dir, scenario_tag, pat_name)
                os.makedirs(save_dir, exist_ok=True)
            
            patient_slices = glob.glob(os.path.join(patient, "*.npy"))
            for patient_slice in patient_slices:
                i = patient_slice.split("/")[-1].split(".")[0]
                data = np.load(patient_slice)
                x_test_r = torch.from_numpy(data).unsqueeze(0).cuda()
                if x_test_r.shape[-1] != 256:
                    x_test_r = resize(x_test_r, (256, 256)).detach()
                x_test_z = x_test_r.clone().cuda().type(torch.cuda.FloatTensor)
                for idx_, k in enumerate(scenario):
                    if k == 0:
                        x_test_z[:, idx_, ...] = impute_tensor
                G_result = generator(x_test_z)
                outputs = G_result.detach().cpu().numpy()[0, scenario==0, :]
                
                inputs = x_test_r.detach().cpu().numpy()[0, scenario==1, :] 
                targets = x_test_r.detach().cpu().numpy()[0, scenario==0, :] 
                # save images
                if vis_dir is not None:
                    save_image(make_image_grid(inputs), f'{save_dir}/{i}_input.png')
                    save_image(make_image_grid(outputs), f'{save_dir}/{i}_output.png')
                    save_image(make_image_grid(targets), f'{save_dir}/{i}_gt.png')
                
               
                for j in range(outputs.shape[0]):
                    img_o = outputs[j, :, :]
                    img_t = targets[j, :, :]
                    
                    img_o = cv2.resize(img_o, (192, 160))
                    img_t = cv2.resize(img_t, (192, 160))
                    
                    img_max = img_t.max()
                    img_o /= img_max
                    img_t /= img_max
                    img_o = img_o.clip(0, 1)
                    
                    metrics_meters[j][0].update(mse(img_o, img_t))
                    metrics_meters[j][1].update(mae(img_o, img_t))
                    metrics_meters[j][2].update(ssim(img_o, img_t))
                    metrics_meters[j][3].update(psnr(img_t, img_o))
    metrics = [{} for _ in range(np.sum(scenario==0))]
    for i in range(np.sum(scenario==0)):
        metrics[i]['mse'] = metrics_meters[i][0].avg
        metrics[i]['mae'] = metrics_meters[i][1].avg
        metrics[i]['ssim'] = metrics_meters[i][2].avg
        metrics[i]['psnr'] = metrics_meters[i][3].avg
    print(f"***{scenario} {metrics}")
    return metrics
                

def evaluator_brats(model, scenarios, patient_list, model_path, vis_dir=None):
    metrics = []
    for scenario in scenarios:
        print(f"***Scenario: {scenario}")
        metric = eval_brats(model, scenario, patient_list, model_path, vis_dir=vis_dir)
        metrics.append(metric)
    return metrics