#!/bin/bash

logs_dir=../logs
checkpoints_dir=../checkpoints
predictions_dir=../predictions

id=$1
data_dir=$2

checkpoints=$(find ${checkpoints_dir} -name "*$id*.checkpoint")
logs=$(find ${logs_dir} -name "*$id*.out")

for log in ${logs[@]} ; do
	id_full=$(echo $(basename ${log}) | sed 's/log_//' | sed 's/.out//')
	cmd=$(grep "train.py" $log | head -1)
	cmd="${cmd} --data_dir ${data_dir} --predict ${predictions_dir}/${id_full}"
	echo $cmd
	python $cmd
done
