from modules.advanced_gans.models import *
from prep_IXI.helpers import create_dataloaders, Resize, ToTensor
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import itertools
import torch
import os
import torch
import argparse
import pdb

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
        image = torch.from_numpy(image_array[i]).unsqueeze(0)
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


def eval_ixi(generator, scenario, patient_list, model_path, vis_dir='vis'):
    scenario_tag = list2str(scenario)
    scenario = np.array(scenario)
    generator.eval()
    impute_tensor = torch.zeros((1, 256, 256), device='cuda')
    metrics_meters = [[AverageMeter() for _ in range(4)] for _ in range(np.sum(scenario==0))]
    # pdb.set_trace()
    with torch.no_grad():
        for patient in tqdm(patient_list):
            pat_name = patient['name'].decode('UTF-8')
            if pat_name == 'IXI014-HH-1236':
                pass
            else:
                print(f'Scenario {scenario_tag}: evaluating {pat_name}')
                patient_image = patient['image']
                patient_copy = patient['image'].clone()
                patient_numpy = patient_copy.detach().cpu().numpy()
                save_dir = os.path.join(model_path, vis_dir, scenario_tag, pat_name)
                os.makedirs(save_dir, exist_ok=True)
                for i in range(patient_numpy.shape[0]):
                    x_test_r = patient_image[i:i+1, ...].cuda()
                    x_test_z = x_test_r.clone().cuda().type(torch.cuda.FloatTensor)
                    for idx_, k in enumerate(scenario):
                        if k == 0:
                            x_test_z[:, idx_, ...] = impute_tensor
                    G_result = generator(x_test_z)
                    outputs = G_result.detach().cpu().numpy()[0, scenario==0, :]
                    inputs = patient_numpy[i, scenario==1, :] 
                    targets = patient_numpy[i, scenario==0, :] 
                    # save images
                    save_image(make_image_grid(inputs), f'{save_dir}/{i}_input.png')
                    save_image(make_image_grid(outputs), f'{save_dir}/{i}_output.png')
                    save_image(make_image_grid(targets), f'{save_dir}/{i}_gt.png')

                    for j in range(outputs.shape[0]):
                        img_o = outputs[j, :, :]
                        img_t = targets[j, :, :]
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
                

def evaluator_ixi(model, scenarios, patient_list, model_path, vis_dir='vis'):
    metrics = []
    for scenario in scenarios:
        print(f"***Scenario: {scenario}")
        metric = eval_ixi(model, scenario, patient_list, model_path, vis_dir=vis_dir)
        metrics.append(metric)
    return metrics



parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--n_contrast', type=int, default=3)
parser.add_argument('--n_slices', type=int, default=90)
parser.add_argument('--n_cases', type=int, default=30)
parser.add_argument('--single', action='store_true', help='single missing data')
parser.add_argument('--ckpt', type=str, default="model/MMGAN/mmgan_ixi_single/generator_param_mmgan_ixi_single_60.pkl")
parser.add_argument('--data_path', type=str, default="/mnt/raid/jiang/projects/SubtleGAN/data/IXI/hdf5/")
parser.add_argument('--model_path', type=str, default="model/MMGAN/mmgan_ixi_single/")
parser.add_argument('--vis_dir', type=str, default="vis")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
generator = GeneratorUNet(in_channels=args.n_contrast, out_channels=args.n_contrast, with_relu=True, with_tanh=False)
ckpt = torch.load(args.ckpt, map_location='cpu')
generator = nn.DataParallel(generator.cuda())
generator.load_state_dict(ckpt['state_dict'])

n_dataloader, dataloader_for_viz = create_dataloaders(parent_path=args.data_path,
                               parent_name='preprocessed',
                               dataset_name='validation_data',
                               dataset_type='cropped',
                               load_pat_names=True,
                               load_seg=False,
                               transform_fn=[Resize(size=(256,256)), ToTensor()],
                               apply_normalization=True,
                               which_normalization=None,
                               resize_slices=args.n_slices,
                               num_workers=4,
                               load_indices=None,
                               dataset="BRATS2018",
                               shuffle=False)
test_patient = []
for k in range(0, args.n_cases):
    test_patient.append(dataloader_for_viz.getitem_via_index(k)) 

if args.single:
    scenarios = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
else:
    scenarios = list(map(list, itertools.product([0, 1], repeat=3)))
    scenarios.remove([0,0,0])
    scenarios.remove([1,1,1])
    
metrics = evaluator_ixi(generator, scenarios, test_patient, args.model_path, vis_dir=args.vis_dir)
with open(os.path.join(args.model_path, args.vis_dir, "test_results.txt"), "w") as f:
     for i, scenario in enumerate(scenarios):
        scenario_tag = list2str(scenario)
        for j in range(len(metrics[i])):
            for m in ['ssim', 'mae', 'psnr', 'mse']:
                msg = f'test_{scenario_tag}/{m}_{j}: {metrics[i][j][m]}\n'
                f.write(msg)
                print(msg)
