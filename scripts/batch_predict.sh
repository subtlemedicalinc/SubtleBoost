export GPU=${GPU:=0}
export DATA_DIR=${DATA_DIR:="/raid/jon/data_full/data"}
export DATA_LIST=${DATA_LIST:="/home/subtle/jon/dev/SubtleGad/data_lists/data_val.txt"}
export LOG_DIR=${LOG_DIR:="/raid/jon/logs"}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:="/raid/jon/checkpoints"}
export PREDICT_DIR=${PREDICT_DIR:="/raid/jon/predictions"}
export MAX_DATA_SETS=${MAX_DATA_SETS:=2}


export JOB_ID=$1

for id in `ls -tr ${CHECKPOINT_DIR}/80c553_* | xargs -n1 basename | sed -e 's/\.checkpoint//g'` ; do
	checkpoint_file="${CHECKPOINT_DIR}/${id}.checkpoint"
	predict_file="${PREDICT_DIR}/${id}"
	if grep --quiet "learn_residual" ${LOG_DIR}/log_${id}.out ; then
		learn_residual_str=" --learn_residual "
	else
		learn_residual_str=" "
	fi

	if grep --quiet "batch_norm" ${LOG_DIR}/log_${id}.out ; then
		batch_norm_str=" --batch_norm "
	else
		batch_norm_str=" "
	fi
	
	python train.py --data_dir ${DATA_DIR} --data_list ${DATA_LIST} --file_ext npy ${learn_residual_str} ${batch_norm_str} --use_multiprocessing --num_workers 4 --max_queue_size 4 --verbose --max_data_sets ${MAX_DATA_SETS} --batch_size 1 --slices_per_input 5 --gpu ${GPU} --checkpoint ${checkpoint_file} --predict ${predict_file}  --id ${id}
done
