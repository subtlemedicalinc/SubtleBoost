#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train_mmgan_brats2018.py --grade=HGG --train_patient_idx=1123 --test_pats=63 --batch_size=4 --dataset=BRATS2018 --n_epochs=60 --model_name=mmgan_hgg_zeros_cl_random --log_level=info --n_cpu=4 --c_learning=1 --z_type=zeros --path_prefix /mnt/raid/jiang/projects/SubtleGAN/data/brats2021/hdf5/
