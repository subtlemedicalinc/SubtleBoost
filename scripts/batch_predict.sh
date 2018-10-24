export GPU=${GPU:=0}
export DATA_DIR=${DATA_DIR:="/raid/jon/data_full/data"}
export DATA_LIST=${DATA_LIST:="/home/subtle/jon/dev/SubtleGad/data_lists/data_val.txt"}
export DATA_NUMBER=${DATA_NUMBER:="-1"}
export LOG_DIR=${LOG_DIR:="/raid/jon/logs"}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:="/raid/jon/checkpoints"}
export PREDICT_DIR=${PREDICT_DIR:="/raid/jon/predictions"}
export MAX_DATA_SETS=${MAX_DATA_SETS:=2}

if [[ "${DATA_NUMBER}" != "-1" ]] ; then
	DATA_LIST="data_list_${DATA_NUMBER}.txt"
	echo "Patient_${DATA_NUMBER}" > ${DATA_LIST}
fi


export JOB_ID=$1

for id in `ls -tr ${CHECKPOINT_DIR}/${JOB_ID}_* | xargs -n1 basename | sed -e 's/\.checkpoint//g'` ; do
	checkpoint_file="${CHECKPOINT_DIR}/${id}.checkpoint"
	predict_file="${PREDICT_DIR}/${id}"
	if grep --quiet "residual_mode=True" ${LOG_DIR}/log_${id}.out ; then
		learn_residual_str=" --learn_residual "
	else
		learn_residual_str=" "
	fi

	if grep --quiet "batch_norm=True" ${LOG_DIR}/log_${id}.out ; then
		batch_norm_str=" --batch_norm "
	else
		batch_norm_str=" "
	fi
	
	python train.py --data_dir ${DATA_DIR} --data_list ${DATA_LIST} --file_ext npy ${learn_residual_str} ${batch_norm_str} --use_multiprocessing --num_workers 4 --max_queue_size 4 --verbose --max_data_sets ${MAX_DATA_SETS} --batch_size 1 --slices_per_input 5 --gpu ${GPU} --checkpoint ${checkpoint_file} --predict ${predict_file}  --id ${id}
done


if [[ "${DATA_NUMBER}" != "-1" ]] ; then
	rm -f  ${DATA_LIST}
fi
