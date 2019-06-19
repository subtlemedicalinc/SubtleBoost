#data_list=data_lists/data_test_tiantan.txt
data_list=data_lists/data_val_tiantan_20190612.txt
checkpoint_file=/raid/jon/checkpoints/b99220_fa31da_epoch50.checkpoint # tiantan model, with no pre-processing except for a global scale factor
path_base=/home/subtle/Data/Tiantan # path to dicom directories
path_out=/raid/jon/predictions/Tiantan_model_b99220_fa31da_epoch50_2019_06_12 # inference output dicoms
gpu=0
#description="TiantanModelNoScale" # append to dicom description
series=998 # series number
description="TiantanModel"


zoom=" "

cat ${data_list} | xargs -n1 -I{} python inference.py --path_base ${path_base}/{} --path_out ${path_out}/{} --verbose --slices_per_input 7 --num_channel_first 32 --gpu ${gpu} --checkpoint ${checkpoint_file} --discard_start_percent 0 --discard_end_percent 0 --normalize --normalize_fun mean --mask_threshold .05 --nslices 50 --transform_type rigid --skip_scale_im --joint_normalize --global_scale_ref_im0 --description ${description} --series_num ${series} ${zoom} --predict /raid/jon/test

cat ${data_list} | xargs -n1 -I{} python scripts/get_raw_dicoms.py --path_base ${path_base}/{} --path_out ${path_out}/{} --override_dicom_naming
