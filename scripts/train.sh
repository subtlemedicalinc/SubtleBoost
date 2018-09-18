#!/bin/bash

commit=$1
	
DATA_DIR=${DATA_DIR:="/local/ubuntu/jon/dev/data_full/data"}
DATA_LIST=${DATA_LIST:="/home/ubuntu/jon/dev/data_full/data_train.txt"}
LEARNING_RATE=${LEARNING_RATE:=".001"}
GPU=${GPU:=0}
MAX_DATA_SETS=${MAX_DATA_SETS:=20}
BATCH_SIZE=${BATCH_SIZE:=8}
NUM_EPOCHS=${NUM_EPOCHS:=20}
NUM_WORKERS=${NUM_WORKERS:=1}
QUEUE_SIZE=${QUEUE_SIZE:=1}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:=0}
SHUFFLE=${SHUFFLE:=1}
BATCH_NORM=${BATCH_NORM:=0}
LEARN_RESIDUAL=${LEARN_RESIDUAL:=0}
SPLIT=${SPLIT:=0}
SLICES_PER_INPUT=${SLICES_PER_INPUT:=1}
FILE_EXT=${FILE_EXT:="h5"}
RANDOM_SEED=${RANDOM_SEED:=723}
VAL_SPLIT=${VAL_SPLIT:="0."}
LOG_DIR=${LOG_DIR:="/local/logs"}
TB_DIR=${TB_DIR:="/local/logs_tb"}
HIST_DIR=${HIST_DIR:="/local/history"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:="/local/checkpoints"}

if [[ ${LEARN_RESIDUAL} -eq "0" ]] ; then
	learn_residual_str=" "
else
	learn_residual_str="--learn_residual"
fi

if [[ ${SPLIT} -eq "0" ]] ; then
	split_str=" "
else
	split_str="--gen_type split"
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



cmd="python train.py --data_dir ${DATA_DIR} --data_list ${DATA_LIST} --file_ext ${FILE_EXT} ${steps_per_epoch_str} ${shuffle_str} ${batch_norm_str} ${learn_residual_str} ${split_str} --num_workers ${NUM_WORKERS} --max_queue_size ${QUEUE_SIZE} --verbose --max_data_sets ${MAX_DATA_SETS} --batch_size ${BATCH_SIZE} --validation_split ${VAL_SPLIT} --learning_rate ${LEARNING_RATE} --slices_per_input ${SLICES_PER_INPUT} --random_seed ${RANDOM_SEED}"

job_id=$(echo $cmd | sha1sum | awk '{print $1}' | cut -c1-6)

checkpoint_file="${commit}_${job_id}.checkpoint"
log_file="log_${commit}_${job_id}.out"
history_file="history_${commit}_${job_id}.npy"

python train.py --data_dir ${DATA_DIR} --data_list ${DATA_LIST} --file_ext ${FILE_EXT} ${steps_per_epoch_str} ${shuffle_str} ${batch_norm_str} ${learn_residual_str} ${split_str} --num_workers ${NUM_WORKERS} --max_queue_size ${QUEUE_SIZE} --verbose --max_data_sets ${MAX_DATA_SETS} --batch_size ${BATCH_SIZE} --validation_split ${VAL_SPLIT} --learning_rate ${LEARNING_RATE} --slices_per_input ${SLICES_PER_INPUT} --random_seed ${RANDOM_SEED} --checkpoint ${CHECKPOINT_DIR}/${checkpoint_file} --log_dir ${TB_DIR} --history_file ${HIST_DIR}/${history_file} --id ${job_id} > ${LOG_DIR}/${log_file} 2>&1
