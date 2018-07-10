#!/bin/bash

commit=$1
LEARNING_RATE=$2
GPU=$3

MAX_DATA_SETS=${MAX_DATA_SETS:=20}
BATCH_SIZE=${BATCH_SIZE:=8}
NUM_EPOCHS=${NUM_EPOCHS:=20}


cmd="py3jon train.py --batch_size ${BATCH_SIZE} --learn_residual --validation_split 0. --learning_rate ${LEARNING_RATE}"
job_id=$(echo $cmd | sha1sum | awk '{print $1}' | cut -c1-6)

checkpoint_file="${commit}_${job_id}.checkpoint"
log_file="log_${commit}_${job_id}.out"

py3jon train.py --data_dir ../data_full/data --verbose --checkpoint ../checkpoints/${checkpoint_file} --num_epochs ${NUM_EPOCHS} --log_dir ../logs_tb --max_data_sets ${MAX_DATA_SETS} --batch_size ${BATCH_SIZE} --gpu $GPU --learn_residual --validation_split 0. --learning_rate ${LEARNING_RATE} > ../logs/${log_file} 2>&1
