#!/bin/sh
split="$1"
python compute_attn_score.py --model_path /mnt/raid/jiang/projects/SubtleGAN/model_fluidstack/MMT/MMTUNetHybrid_GAN_ls_zerogad_s_c_MMT_epo50_bs32_lrg0.0005_192_opt-adamw --ckpt epoch_31.pth --head_norm --zero_gad --split $split
python compute_attn_score.py --model_path /mnt/raid/jiang/projects/SubtleGAN/model/MMT/MMTUNetHybrid_GAN_s_c_MMT_epo50_bs24_lrg0.0005_192_opt-adamw/ --ckpt epoch_49.pth --head_norm --k 3 --split $split
python compute_attn_score.py --model_path /mnt/raid/jiang/projects/SubtleGAN/model_fluidstack/MMT/MMTUNetHybrid_GAN_ls_random_s_c_MMT_epo75_bs32_lrg0.0005_192_opt-adamw/ --ckpt epoch_74.pth --head_norm --k 3 --split $split
python compute_attn_score.py --model_path /mnt/raid/jiang/projects/SubtleGAN/model_fluidstack/MMT/MMTUNetHybrid_GAN_ls_zerogad_s_c_MMT_epo50_bs32_lrg0.0005_192_opt-adamw --ckpt epoch_31.pth --zero_gad --split $split
python compute_attn_score.py --model_path /mnt/raid/jiang/projects/SubtleGAN/model/MMT/MMTUNetHybrid_GAN_s_c_MMT_epo50_bs24_lrg0.0005_192_opt-adamw/ --ckpt epoch_49.pth --k 3 --split $split
python compute_attn_score.py --model_path /mnt/raid/jiang/projects/SubtleGAN/model_fluidstack/MMT/MMTUNetHybrid_GAN_ls_random_s_c_MMT_epo75_bs32_lrg0.0005_192_opt-adamw/ --ckpt epoch_74.pth --k 3 --split $split
