#!/bin/bash

commit=${commit:=$(git rev-parse HEAD | cut -c1-6)}
	


PREDICT_DIR=${PREDICT_DIR:="/raid/jon/predictions/dicoms"}
DATA_RAW=${DATA_RAW:="/home/subtle/Data/Stanford/lowcon"}
DATA_DIR=${DATA_DIR:="/local/ubuntu/jon/dev/data_full/data"}
DATA_LIST=${DATA_LIST:="/home/ubuntu/jon/dev/data_full/data_train.txt"}
DATA_LIST_TEST=${DATA_LIST_TEST:="/home/ubuntu/jon/dev/data_full/data_test.txt"}
LEARNING_RATE=${LEARNING_RATE:=".001"}
GPU=${GPU:=0}
MAX_DATA_SETS=${MAX_DATA_SETS:=20}
BATCH_SIZE=${BATCH_SIZE:=8}
NUM_EPOCHS=${NUM_EPOCHS:=20}
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



cmd="python train.py --data_dir ${DATA_DIR} --data_list ${DATA_LIST} --file_ext ${FILE_EXT} ${steps_per_epoch_str} ${val_steps_per_epoch_str} ${shuffle_str} ${batch_norm_str} ${learn_residual_str} ${positive_only_str} ${split_str} ${multiprocessing_str} ${no_save_best_only_str} --num_epochs ${NUM_EPOCHS} --num_workers ${NUM_WORKERS} --max_queue_size ${QUEUE_SIZE} --verbose --max_data_sets ${MAX_DATA_SETS} --batch_size ${BATCH_SIZE} --validation_split ${VAL_SPLIT} --learning_rate ${LEARNING_RATE} --slices_per_input ${SLICES_PER_INPUT} --random_seed ${RANDOM_SEED} --l1_lambda ${L1_LAMBDA} --ssim_lambda ${SSIM_LAMBDA} --num_channel_first ${NUM_CHANNEL_FIRST}"

job_id=$(echo $cmd | sha1sum | awk '{print $1}' | cut -c1-6)

checkpoint_file="${commit}_${job_id}.checkpoint"
log_file="log_inference_${commit}_${job_id}.out"
out_dir=${PREDICT_DIR}/${commit}_${job_id}

mkdir -p ${out_dir}

cat ${DATA_LIST_TEST} | xargs -n1 -I{} python inference.py --data_preprocess ${DATA_DIR}/{}.${FILE_EXT} --path_base ${DATA_RAW}/{} --path_out ${out_dir}/{}/{}_SubtleGad_rehist ${batch_norm_str} ${learn_residual_str} ${positive_only_str} ${split_str} ${multiprocessing_str} --num_workers ${NUM_WORKERS} --max_queue_size ${QUEUE_SIZE} --verbose --slices_per_input ${SLICES_PER_INPUT} --num_channel_first ${NUM_CHANNEL_FIRST} --gpu ${GPU} --checkpoint ${CHECKPOINT_DIR}/${checkpoint_file} --log_dir ${TB_DIR} --id ${job_id} > ${LOG_DIR}/${log_file} 2>&1
