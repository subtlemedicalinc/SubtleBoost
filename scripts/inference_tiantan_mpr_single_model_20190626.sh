#data_list=data_lists/data_test_tiantan.txt
data_list=data_lists/data_val_tiantan_20190612.txt
checkpoint_dir=/raid/jon/checkpoints
path_base=/home/subtle/Data/Tiantan # path to dicom directories
path_out=/raid/jon/predictions/Tiantan_mpr_single_model_b97aa9 # inference o utput dicoms
gpu=2
series=899
description="MedSingleModel"
avg_mode="median"

checkpoint_file=${checkpoint_dir}/b97aa9_958d27.checkpoint

cat ${data_list} | xargs -n1 -I{} python inference.py --path_base ${path_base}/{} --path_out ${path_out}/{}_${description} --verbose --slices_per_input 7 --resize 240 --num_channel_first 32 --gpu ${gpu} --discard_start_percent 0 --discard_end_percent 0 --normalize --normalize_fun mean --mask_threshold .05 --nslices 50 --transform_type rigid --skip_scale_im --joint_normalize --global_scale_ref_im0 --description ${description} --series_num ${series} --checkpoint ${checkpoint_file} --inference_mpr --inference_mpr_avg ${avg_mode}
