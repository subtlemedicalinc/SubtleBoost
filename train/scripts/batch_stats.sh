#!/bin/bash

export DATA_DIR=${DATA_DIR:="/raid/jon/data_full/data"}
export DATA_LIST=${DATA_LIST:="/home/subtle/jon/dev/SubtleGad/data_lists/data_val.txt"}
export PREDICT_DIR=${PREDICT_DIR:="/raid/jon/predictions"}

export JOB_ID=$1

for id in `ls -tr ${PREDICT_DIR} | grep ${JOB_ID} | xargs -n1 basename` ; do
	#predict_file="${PREDICT_DIR}/${id}"
	#echo ${id}
	cd ${PREDICT_DIR}/${id}
	for pat in `ls -tr | grep "Patient" | grep "npy"` ; do
		patient=$(echo $pat | sed -e 's/_predict.*\.npy//')
		truth_file="${DATA_DIR}/${patient}.npy"
		prediction_file="${PREDICT_DIR}/${id}/${pat}"
		out_file="${PREDICT_DIR}/${id}/${patient}_${id}_stats.h5"
		echo "ID: ${id} Patient: ${patient}"
		python /home/subtle/jon/dev/SubtleGad/compute_stats.py --all_slices --truth ${truth_file} --prediction ${prediction_file} --output ${out_file} --cutoff .05
	done
	cd -
	
done
