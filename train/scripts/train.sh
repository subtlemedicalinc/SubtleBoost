#!/bin/bash

function show_usage() {
	echo "Usage: $0 experiment_name log_dir"
}

function show_help() {
	echo ""
	echo "SubtleGad train convenience script."
	echo ""
}

OPTIND=1

if [[ $# -ne 2 ]] ; then
	show_usage >&2
	exit 1
fi

while getopts "h?" opt; do
	case "${opt}" in
		h)
			show_usage >&1
			show_help >&1
			exit 0
			;;
		\?)
			show_usage >&2
			exit 1
			;;
	esac
done

shift "$((OPTIND-1))"

commit=${commit:=$(git rev-parse HEAD | cut -c1-6)}
exparg=$1

if [[ $exparg == *"/"* ]]; then
  expname="$(cut -d'/' -f1 <<<"$exparg")"
  subexp="$(cut -d'/' -f2 <<<"$exparg")"
  exp_str="--experiment ${expname} --sub_experiment ${subexp}"
else
  expname="${exparg}"
  exp_str="--experiment ${exparg}"
fi

if [[ $GPU == '' ]]; then
  gpu_str=""
else
  gpu_str="--gpu ${GPU}"
fi

if test -f configs/experiments/${expname}/config.json; then
  echo ""
else
  echo "Invalid experiment name - ${expname}"
  exit 1
fi

fcontent=$(python3 ./scripts/utils/print_config_json.py configs/experiments/${expname}/config.json train)
fcontent=${fcontent}--${exparg}

job_id=$(echo ${fcontent} | sha1sum | awk '{print $1}' | cut -c1-6)
logfile=$2/log_train_${commit}_${job_id}.log

python train_process.py ${exp_str} ${gpu_str} --id ${job_id}
