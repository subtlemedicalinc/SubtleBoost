data_list=data_lists/data_test_tiantan_20190711.txt
checkpoint_dir=/raid/jon/checkpoints
job_id=02859f_11e2d9
path_base=/home/subtle/Data/Tiantan # path to dicom directories
path_out=/raid/jon/predictions/Tiantan_mpr_rot5_${job_id}_20190711 # inference o utput dicoms
gpu=2
series=1001
description='SubtleGad'
avg_mode='mean'
num_rot=5

checkpoint_file=${checkpoint_dir}/${job_id}.checkpoint

cat ${data_list} | xargs -n1 -I{} python inference.py --path_base ${path_base}/{} --path_out ${path_out}/{}_${description} --verbose --slices_per_input 7 --resize 240 --num_channel_first 32 --gpu ${gpu} --discard_start_percent 0 --discard_end_percent 0 --normalize --normalize_fun mean --mask_threshold .05 --nslices 50 --transform_type rigid --skip_scale_im --joint_normalize --global_scale_ref_im0 --description ${description} --series_num ${series} --checkpoint ${checkpoint_file} --inference_mpr --inference_mpr_avg ${avg_mode} --num_rotations ${num_rot} --override_dicom_naming --stats_file ${path_out}/stats_{}_${description}.h5
