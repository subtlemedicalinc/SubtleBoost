#!/bin/bash

commit=${commit:=$(git rev-parse HEAD | cut -c1-6)}

DATA_DIR=${DATA_DIR:="/local/ubuntu/jon/dev/data_full/data"}
DATA_LIST=${DATA_LIST:="/home/ubuntu/jon/dev/data_full/data_train.txt"}
LEARNING_RATE=${LEARNING_RATE:=".001"}
GPU=${GPU:=0}
MAX_DATA_SETS=${MAX_DATA_SETS:=20}
BATCH_SIZE=${BATCH_SIZE:=8}
NUM_EPOCHS=${NUM_EPOCHS:=100}
NUM_WORKERS=${NUM_WORKERS:=1}
QUEUE_SIZE=${QUEUE_SIZE:=1}
MULTIPROCESSING=${MULTIPROCESSING:=0}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:=0}
VAL_STEPS_PER_EPOCH=${VAL_STEPS_PER_EPOCH:=0}
NUM_CHANNEL_FIRST=${NUM_CHANNEL_FIRST:=32}
SHUFFLE=${SHUFFLE:=1}
BATCH_NORM=${BATCH_NORM:=0}
LEARN_RESIDUAL=${LEARN_RESIDUAL:=0}
POSITIVE_ONLY=${POSITIVE_ONLY:=0}
SPLIT=${SPLIT:=0}
SLICES_PER_INPUT=${SLICES_PER_INPUT:=1}
FILE_EXT=${FILE_EXT:="h5"}
RANDOM_SEED=${RANDOM_SEED:=723}
VAL_SPLIT=${VAL_SPLIT:="0."}
LOG_DIR=${LOG_DIR:="/local/logs"}
TB_DIR=${TB_DIR:="/local/logs_tb"}
HIST_DIR=${HIST_DIR:="/local/history"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:="/local/checkpoints"}
L1_LAMBDA=${L1_LAMBDA:="1."}
SSIM_LAMBDA=${SSIM_LAMBDA:="0."}
NO_SAVE_BEST_ONLY=${NO_SAVE_BEST_ONLY:=0}
TRAIN_MPR=${TRAIN_MPR:=0}
RESIZE=${RESIZE:=0}
SLICE_AXIS=${SLICE_AXIS:=0}
RESAMPLE_SIZE=${RESAMPLE_SIZE:=0}

if [[ ${TRAIN_MPR} -eq "0" ]] ; then
	train_mpr_str=" "
else
	train_mpr_str="--train_mpr"
fi

if [[ ${RESIZE} -eq "0" ]] ; then
	resize_str=" "
else
	resize_str="--resize ${RESIZE}"
fi

if [[ ${NO_SAVE_BEST_ONLY} -eq "0" ]] ; then
	no_save_best_only_str=" "
else
	no_save_best_only_str="--no_save_best_only"
fi

if [[ ${LEARN_RESIDUAL} -eq "0" ]] ; then
	learn_residual_str=" "
else
	learn_residual_str="--learn_residual"
fi

if [[ ${POSITIVE_ONLY} -eq "0" ]] ; then
	positive_only_str=" "
else
	positive_only_str="--positive_only"
fi

if [[ ${MULTIPROCESSING} -eq "0" ]] ; then
	multiprocessing_str=" "
else
	multiprocessing_str="--use_multiprocessing"
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

if [[ ${VAL_STEPS_PER_EPOCH} -eq "0" ]] ; then
	val_steps_per_epoch_str=" "
else
	val_steps_per_epoch_str="--val_steps_per_epoch ${VAL_STEPS_PER_EPOCH}"

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

if [[ ${RESAMPLE_SIZE} -eq "0" ]] ; then
	resample_size=" "
else
	resample_size="--resample_size ${RESAMPLE_SIZE}"

fi



cmd="python train.py --data_dir ${DATA_DIR} --data_list ${DATA_LIST} --file_ext ${FILE_EXT} ${steps_per_epoch_str} ${val_steps_per_epoch_str} ${shuffle_str} ${batch_norm_str} ${learn_residual_str} ${positive_only_str} ${split_str} ${multiprocessing_str} ${no_save_best_only_str} ${resize_str} ${train_mpr_str} --slice_axis ${SLICE_AXIS} --num_epochs ${NUM_EPOCHS} --num_workers ${NUM_WORKERS} --max_queue_size ${QUEUE_SIZE} --verbose --max_data_sets ${MAX_DATA_SETS} --batch_size ${BATCH_SIZE} --validation_split ${VAL_SPLIT} --learning_rate ${LEARNING_RATE} --slices_per_input ${SLICES_PER_INPUT} --random_seed ${RANDOM_SEED} --l1_lambda ${L1_LAMBDA} --ssim_lambda ${SSIM_LAMBDA} --num_channel_first ${NUM_CHANNEL_FIRST}"

job_id=$(echo $cmd | sha1sum | awk '{print $1}' | cut -c1-6)

checkpoint_file="${commit}_${job_id}.checkpoint"

log_file="log_${commit}_${job_id}.out"
history_file="history_${commit}_${job_id}.npy"

python train.py --data_dir ${DATA_DIR} --data_list ${DATA_LIST} --file_ext ${FILE_EXT} ${steps_per_epoch_str} ${val_steps_per_epoch_str} ${shuffle_str} ${batch_norm_str} ${learn_residual_str} ${positive_only_str} ${split_str} ${multiprocessing_str} ${resample_size} ${no_save_best_only_str} ${resize_str} ${train_mpr_str} --slice_axis ${SLICE_AXIS} --num_epochs ${NUM_EPOCHS} --num_workers ${NUM_WORKERS} --max_queue_size ${QUEUE_SIZE} --verbose --max_data_sets ${MAX_DATA_SETS} --batch_size ${BATCH_SIZE} --validation_split ${VAL_SPLIT} --learning_rate ${LEARNING_RATE} --slices_per_input ${SLICES_PER_INPUT} --random_seed ${RANDOM_SEED} --l1_lambda ${L1_LAMBDA} --ssim_lambda ${SSIM_LAMBDA} --num_channel_first ${NUM_CHANNEL_FIRST} --gpu ${GPU} --checkpoint ${CHECKPOINT_DIR}/${checkpoint_file} --log_dir ${TB_DIR} --history_file ${HIST_DIR}/${history_file} --id ${job_id} > ${LOG_DIR}/${log_file} 2>&1
