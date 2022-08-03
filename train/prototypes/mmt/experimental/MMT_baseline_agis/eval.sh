#sCUDA_VISIBLE_DEVICES=7 python test.py --n_gpu 1 --ckpt epoch_99.pth --model_path /mnt/raid/jiang/projects/SubtleGAN/model/MMT/MMTBaseline_agis_random_MMT_epo100_bs16_lrg0.0001_192 --vis 

CUDA_VISIBLE_DEVICES=7 python test.py --n_gpu 1 --ckpt epoch_99.pth --model_path /mnt/raid/jiang/projects/SubtleGAN/MMT/model/MMT/IXI_Baseline_agis_random_MMT_epo100_bs16_lrg0.0001_192 --data_path /mnt/raid/jiang/projects/SubtleGAN/data/IXI/IXI_slices --dataset IXI --n_contrast 3 --vis

#CUDA_VISIBLE_DEVICES=7 python test.py --n_gpu 1 --ckpt epoch_99.pth --model_path /mnt/raid/jiang/projects/SubtleGAN/model/MMT/MMTBaseline_agis_MMT_epo100_bs16_lrg0.0001_192/ --vis --k 3

CUDA_VISIBLE_DEVICES=7 python test.py --n_gpu 1 --ckpt epoch_99.pth --model_path /mnt/raid/jiang/projects/SubtleGAN/MMT/model/MMT/IXI_Baseline_agis_single_MMT_epo100_bs16_lrg0.0001_192 --data_path /mnt/raid/jiang/projects/SubtleGAN/data/IXI/IXI_slices --dataset IXI --k 2 --n_contrast 3 --vis


