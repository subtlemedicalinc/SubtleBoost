#data_list=data_lists/data_test_tiantan.txt
data_list=data_lists/data_val_tiantan_20190612.txt
checkpoint_dir=/raid/jon/checkpoints
path_base=/home/subtle/Data/Tiantan # path to dicom directories
path_out=/raid/jon/predictions/Tiantan_mpr_a12c1c # inference o utput dicoms
gpu=0
series=904
description="Med"
avg_mode="median"

checkpoint_file_0=${checkpoint_dir}/a12c1c_77b33f.checkpoint
checkpoint_file_2=${checkpoint_dir}/a12c1c_88b8bd.checkpoint
checkpoint_file_3=${checkpoint_dir}/a12c1c_b6d8d1.checkpoint

cat ${data_list} | xargs -n1 -I{} python inference.py --path_base ${path_base}/{} --path_out ${path_out}/{}_${description} --verbose --slices_per_input 7 --resize 240 --num_channel_first 32 --gpu ${gpu} --discard_start_percent 0 --discard_end_percent 0 --normalize --normalize_fun mean --mask_threshold .05 --nslices 50 --transform_type rigid --skip_scale_im --joint_normalize --global_scale_ref_im0 --description ${description} --series_num ${series} --checkpoint_slice_axis_0 ${checkpoint_file_0} --checkpoint_slice_axis_2 ${checkpoint_file_2} --checkpoint_slice_axis_3 ${checkpoint_file_3} --inference_mpr --inference_mpr_avg ${avg_mode}

#description="Sag"
#series=900
#checkpoint_file=${checkpoint_dir}/a12c1c_77b33f.checkpoint

#description="Ax"
#series=901
#gpu=2
#cat ${data_list} | xargs -n1 -I{} python inference.py --path_base ${path_base}/{} --path_out ${path_out}/{}_${description} --verbose --slices_per_input 7 --resize 240 --num_channel_first 32 --gpu ${gpu} --checkpoint ${checkpoint_file} --discard_start_percent 0 --discard_end_percent 0 --normalize --normalize_fun mean --mask_threshold .05 --nslices 50 --transform_type rigid --skip_scale_im --joint_normalize --global_scale_ref_im0 --description ${description} --series_num ${series} --slice_axis 2 &

#description="Cor"
#series=902
#gpu=3
#cat ${data_list} | xargs -n1 -I{} python inference.py --path_base ${path_base}/{} --path_out ${path_out}/{}_${description} --verbose --slices_per_input 7 --resize 240 --num_channel_first 32 --gpu ${gpu} --checkpoint ${checkpoint_file} --discard_start_percent 0 --discard_end_percent 0 --normalize --normalize_fun mean --mask_threshold .05 --nslices 50 --transform_type rigid --skip_scale_im --joint_normalize --global_scale_ref_im0 --description ${description} --series_num ${series} --slice_axis 3 



#cat ${data_list} | xargs -n1 -I{} python scripts/get_raw_dicoms.py --path_base ${path_base}/{} --path_out ${path_out}/{} --override_dicom_naming
