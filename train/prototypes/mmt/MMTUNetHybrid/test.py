import argparse
import os
import glob
import numpy as np
import torch
from networks.mmt import MMT as generator
from torchvision.utils import save_image
from tqdm import tqdm
from configs.config import get_config
from evaluator import evaluator_brats, evaluator_ixi, evaluator_spine, split_data, generate_spine_images
from utils import list2str, make_image_grid


input_combination_brats = [[0], [1], [2], [3], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [0, 1, 2],
                     [0, 1, 3], [0, 2, 3], [1, 2, 3]]

input_combination_3 = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]

input_combination_zerogad = [[0, 2, 3]]

input_combination_ixi = [[0], [1], [2], [0, 1], [0, 2], [1, 2]]

input_combination_2 = [[1, 2], [0, 2], [0, 1]]

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/mnt/raid/jiang/projects/SubtleGAN/data/brats2021_slices_crop192x160', help='root dir for data')
parser.add_argument('--cfg', type=str, default='configs/mmt.yml')
parser.add_argument('--dataset', type=str,
                    default='BRATS', help='experiment_name')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--k', type=int,
                    default=None, help='number of inputs')
parser.add_argument('--zero_gad', action='store_true', help='eval zero_gad')
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--vis', action='store_true', help='visualize results')
parser.add_argument('--seg', action='store_true', help='test seg acc')
parser.add_argument('--masked', action='store_true', help='test similarity within tumor mask')
parser.add_argument('--vis_dir', type=str, default='vis')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_contrast', type=int, default=4, help='total number of contrast in the dataset')

args = parser.parse_args()


def vis_results(model, inputs, targets, files, model_path, split, vis_dir='vis', dataset='BRATS'):
    input_tag = list2str(inputs)
    model.eval()
    with torch.no_grad():
        for file in tqdm(files):
            data = np.load(file)
            n_channel = data.shape[0]
            image = torch.from_numpy(data.astype(np.float32)).unsqueeze(0).cuda()
            data = [image[:, i, :, :].unsqueeze(0) for i in range(n_channel)]  # [(1, 1, H, W)]
            img_inputs, img_targets, contrast_input, contrast_output = split_data(data, inputs, targets)
            if dataset == 'Spine':
                img_outputs = generate_spine_images(data, model, contrast_input, contrast_output)
            else:
                img_outputs, _, _ = model(img_inputs, contrast_input, contrast_output)

            # save results
            case = file.split("/")[-2]
            save_dir = os.path.join(model_path, vis_dir, split, input_tag, case)
            os.makedirs(save_dir, exist_ok=True)
            slice_num = file.split("/")[-1].split(".")[0]
            save_image(make_image_grid(img_inputs), f'{save_dir}/{slice_num}_input.png')
            save_image(make_image_grid(img_outputs), f'{save_dir}/{slice_num}_output.png')
            save_image(make_image_grid(img_targets), f'{save_dir}/{slice_num}_gt.png')
            
            
def visualize_results(args, model, input_combination, split='test', vis_dir='vis'):
    data_dir = os.path.join(args.root_path, split)
    if args.dataset == 'BRATS':
        cases = glob.glob(f"{data_dir}/Bra*")
    elif args.dataset == 'IXI':
        cases = glob.glob(f"{data_dir}/IXI*")
    else:
        cases = glob.glob(f"{data_dir}/*")
    files = []
    # get all slices
    for case in cases:
        files += glob.glob(f'{case}/*.npy')
    print("The length of data set is: {}".format(len(files)))
    for inputs in input_combination:
        targets = list(set(range(args.n_contrast)) - set(inputs))
        print(f"***Inputs: {inputs}; Outputs: {targets}")
        vis_results(model, inputs, targets, files, args.model_path, split, vis_dir=vis_dir, dataset=args.dataset)


if __name__ == "__main__":
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
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                seg=args.seg,
                                num_contrast=args.n_contrast).cuda()

    state_dict = torch.load(os.path.join(args.model_path, args.ckpt), map_location='cpu')
    G.load_state_dict(state_dict['G'])
    G.eval()

    if args.zero_gad:
        input_combination = input_combination_zerogad
    elif args.k == 3:
        input_combination = input_combination_3
    elif args.k == 2:
        input_combination = input_combination_2
    else:
        input_combination = input_combination_brats if args.dataset == 'BRATS' else input_combination_ixi
    if args.vis:
        visualize_results(args, G, input_combination, split='test', vis_dir=args.vis_dir)

    if args.dataset == 'BRATS':
        metrics, metrics_seg, metrics_masked = evaluator_brats(args, G, input_combination, split='test', seg=args.seg,
                                                           masked=args.masked)
    elif args.dataset == 'IXI':
        metrics = evaluator_ixi(args, G, input_combination, split='test')
    else:
        metrics = evaluator_spine(args, G, input_combination, split='test')

    with open(os.path.join(args.model_path, args.vis_dir, "test_results.txt"), "w") as f:
        for i in range(len(input_combination)):
            output_combination = list(set(range(args.n_contrast)) - set(input_combination[i]))
            for j in range(len(metrics[i])):
                for m in ['ssim', 'mae', 'psnr', 'mse']:
                    msg = f'test_{list2str(input_combination[i])}/{m}_{output_combination[j]}: {metrics[i][j][m]}\n'
                    f.write(msg)
                    print(msg)
        if args.masked:
            for i in range(len(input_combination)):
                output_combination = list(set(range(args.n_contrast)) - set(input_combination[i]))
                for j in range(len(metrics[i])):
                    for m in ['masked_mae', 'masked_psnr', 'masked_mse']:
                        msg = f'test_masked{list2str(input_combination[i])}/{m}_{output_combination[j]}: {metrics_masked[i][j][m]}\n'
                        f.write(msg)
                        print(msg)
        if args.seg:
            for i in range(len(input_combination)):
                for m in ['ET', 'TC', 'WT']:
                    msg = f'test_{list2str(input_combination[i])}/seg_{m}: {metrics_seg[i][m]}\n'
                    f.write(msg)
                    print(msg)