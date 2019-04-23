data_list=data_lists/data_test_tiantan_batch1.txt
checkpoint_file=/raid/jon/checkpoints/ac719d_8e99ba.checkpoint # hoag model, with no pre-processing except for a global scale factor
path_base=/home/subtle/Data/Tiantan_Batch1 # path to dicom directories
path_out=/raid/jon/predictions/Tiantan_Batch1_2019_04_23 # inference output dicoms
gpu=1
description="HoagModelNoScale" # append to dicom description
series=999 # series number

#cat ${data_list} | xargs -n1 -I{} python inference.py --path_base ${path_base}/{} --path_out ${path_out}/{} --verbose --slices_per_input 5 --num_channel_first 32 --gpu ${gpu} --checkpoint ${checkpoint_file} --discard_start_percent 0 --discard_end_percent 0 --normalize --normalize_fun mean --transform_type rigid --skip_scale_im --joint_normalize --skip_mask --description ${description} --series_num ${series} --zoom 320

cat ${data_list} | xargs -n1 -I{} python inference.py --path_base ${path_base}/{} --path_out ${path_out}/{} --verbose --slices_per_input 5 --num_channel_first 32 --gpu ${gpu} --checkpoint ${checkpoint_file} --discard_start_percent 0 --discard_end_percent 0 --normalize --normalize_fun mean --transform_type rigid --skip_scale_im --joint_normalize --skip_mask --description ${description} --series_num ${series} --scale_dicom_tags --override_dicom_naming --zoom 320

cat ${data_list} | xargs -n1 -I{} python scripts/get_raw_dicoms.py --path_base ${path_base}/{} --path_out ${path_out}/{} --override_dicom_naming
