#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python train_mmgan_ixi.py --grade=HGG --train_patient_idx=519 --test_pats=28 --batch_size=6 --dataset=BRATS2018 --n_epochs=60 --model_name=mmgan_ixi_random --log_level=info --n_cpu=4 --c_learning=1 --z_type=zeros --path_prefix /mnt/raid/jiang/projects/SubtleGAN/data/IXI/hdf5/
