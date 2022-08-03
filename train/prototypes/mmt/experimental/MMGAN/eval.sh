python evaluate_brats_mmt.py --gpu 1 --n_contrast 4 --model_path model/MMGAN/mmgan_brats_zeros_cl_random_0110 --data_path /mnt/raid/jiang/projects/SubtleGAN/data/brats2021_slices_crop192x160/test --save_dir vis --vis --ckpt model/MMGAN/mmgan_brats_zeros_cl_random_0110/generator_param_mmgan_brats_zeros_cl_random_0110_60.pkl

python evaluate_ixi_mmt.py --gpu 1 --n_contrast 3 --model_path model/MMGAN/mmgan_ixi_random_0118 --data_path /mnt/raid/jiang/projects/SubtleGAN/data/IXI/IXI_slices/test --save_dir vis --vis --ckpt model/MMGAN/mmgan_ixi_random_0118/generator_param_mmgan_ixi_random_0118_60.pkl

python evaluate_brats_mmt.py --gpu 1 --single --n_contrast 4 --ckpt model/MMGAN/mmgan_brats_zeros_cl_single_0110/generator_param_mmgan_brats_zeros_cl_single_0110_60.pkl --model_path model/MMGAN/mmgan_brats_zeros_cl_single_0110 --data_path /mnt/raid/jiang/projects/SubtleGAN/data/brats2021_slices_crop192x160/test --save_dir vis --vis 

python evaluate_ixi_mmt.py --gpu 1 --single --n_contrast 3 --model_path model/MMGAN/mmgan_ixi_single_0118 --data_path /mnt/raid/jiang/projects/SubtleGAN/data/IXI/IXI_slices/test --save_dir vis --vis --ckpt model/MMGAN/mmgan_ixi_single_0118/generator_param_mmgan_ixi_single_0118_60.pkl
