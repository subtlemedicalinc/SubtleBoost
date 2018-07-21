#!/bin/bash

commit=$1
	
LEARNING_RATE=${LEARNING_RATE:=".001"}
GPU=${GPU:=0}
MAX_DATA_SETS=${MAX_DATA_SETS:=20}
BATCH_SIZE=${BATCH_SIZE:=8}
NUM_EPOCHS=${NUM_EPOCHS:=20}
NUM_WORKERS=${NUM_WORKERS:=1}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:=0}
SHUFFLE=${SHUFFLE:=1}
BATCH_NORM=${BATCH_NORM:=0}
LEARN_RESIDUAL=${LEARN_RESIDUAL:=0}

if [[ ${LEARN_RESIDUAL} -eq "0" ]] ; then
	learn_residual_str=" "
else
	learn_residual_str="--learn_residual"

fi
if [[ ${STEPS_PER_EPOCH} -eq "0" ]] ; then
	steps_per_epoch_str=" "
else
	steps_per_epoch_str="--steps_per_epoch ${STEPS_PER_EPOCH}"

fi

if [[ ${SHUFFLE} -eq "0" ]] ; then
	shuffle_str=" "
else
	shuffle_str="--shuffle"

fi

if [[ ${BATCH_NORM} -eq "0" ]] ; then
	batch_norm_str=" "
else
	batch_norm_str="--batch_norm"

fi


cmd="python train.py ${steps_per_epoch_str} ${shuffle_str} ${batch_norm_str} ${learn_residual_str} --batch_size ${BATCH_SIZE} --validation_split 0. --learning_rate ${LEARNING_RATE}"
job_id=$(echo $cmd | sha1sum | awk '{print $1}' | cut -c1-6)

checkpoint_file="${commit}_${job_id}.checkpoint"
log_file="log_${commit}_${job_id}.out"
history_file="history_${commit}_${job_id}.npy"

python train.py --data_dir ../data_full/data ${steps_per_epoch_str} ${shuffle_str} ${batch_norm_str} ${learn_residual_str} --num_workers ${NUM_WORKERS} --verbose --checkpoint ../checkpoints/${checkpoint_file} --num_epochs ${NUM_EPOCHS} --log_dir ../logs_tb --max_data_sets ${MAX_DATA_SETS} --batch_size ${BATCH_SIZE} --gpu $GPU --validation_split 0. --learning_rate ${LEARNING_RATE} --history_file ../history/${history_file} --id ${job_id} > ../logs/${log_file} 2>&1
