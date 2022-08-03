# for single, best ssim and best mae is epoch 40, best psnr is epoch 47, we pick best_mae as the difference in psnr for the two is small
# for random, best ssim and best mae is epoch 47, best psnr is epoch 45, we pick best_mae as the difference in psnr for the two is basically the same
CUDA_VISIBLE_DEVICES=7 python test_new.py --n_gpu 1 --ckpt best_mae.pth --model_path /mnt/raid/jiang/projects/SubtleGAN/MMT/model/IXI_single_new_epo50_bs20_lrg0.0005_s1234_opt-adamw --vis --k 2 --dataset IXI --data_path /mnt/raid/jiang/projects/SubtleGAN/data/IXI/IXI_slices 

CUDA_VISIBLE_DEVICES=7 python test_new.py --n_gpu 1 --ckpt best_mae.pth --model_path /mnt/raid/jiang/projects/SubtleGAN/MMT/model/IXI_random_new_epo50_bs20_lrg0.0005_s1234_opt-adamw --vis --dataset IXI --data_path /mnt/raid/jiang/projects/SubtleGAN/data/IXI/IXI_slices 

CUDA_VISIBLE_DEVICES=7 python test_new.py --n_gpu 1 --ckpt epoch_49.pth --model_path /mnt/raid/jiang/projects/SubtleGAN/model/MMT/MMTUNetHybrid_GAN_s_c_MMT_epo50_bs24_lrg0.0005_192_opt-adamw/ --vis --k 3 --dataset BRATS --data_path /mnt/raid/jiang/projects/SubtleGAN/data/brats2021_slices_crop192x160

CUDA_VISIBLE_DEVICES=7 python test_new.py --n_gpu 1 --ckpt epoch_74.pth --model_path /mnt/raid/jiang/projects/SubtleGAN/model_fluidstack/MMT/MMTUNetHybrid_GAN_ls_random_s_c_MMT_epo75_bs32_lrg0.0005_192_opt-adamw/ --vis --dataset BRATS --data_path /mnt/raid/jiang/projects/SubtleGAN/data/brats2021_slices_crop192x160
