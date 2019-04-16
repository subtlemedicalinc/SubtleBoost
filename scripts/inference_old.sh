#!/bin/bash

### TODO ###
# change job id
# change checkpoint
# change directories
# change network params

GPU=${GPU:=0}
PATH_OUT=${PATH_OUT:=/raid/jon/predictions/dicoms}

patient_name=$1

python inference.py --data_preprocess /raid/jon/data_full_stanford/data/${patient_name}.h5 --path_base /home/subtle/Data/Stanford/lowcon/${patient_name} --path_out ${PATH_OUT}/${patient_name} --verbose --gpu ${GPU} --checkpoint /raid/jon/checkpoints/1bfe8d_c7f660.checkpoint --id 1bfe8d_c7f660 --slices_per_input 5
