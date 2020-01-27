#!/bin/bash

function show_usage() {
	echo "Usage: $0 [-g gpu_str] experiment_name log_dir"
	echo ""
	echo "-g gpu_str: specify gpus to use, e.g. '0,1,3'"
	echo ""
}

function show_help() {
	echo ""
	echo "SubtleGad batch preprocessing convenience script."
	echo ""
}

OPTIND=1

if [[ $# -lt 2 ]] ; then
	show_usage >&2
	exit 1
fi

export GPU=' '

while getopts "h?g:" opt; do
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
		g)
			export GPU="--gpu ${OPTARG}"
			;;
	esac
done

shift "$((OPTIND-1))"

logfile=$2/preprocess.out
errfile=$2/preprocess.err

exparg=$1
if [[ $exparg == *"/"* ]]; then
  expname="$(cut -d'/' -f1 <<<"$exparg")"
  subexp="$(cut -d'/' -f2 <<<"$exparg")"
  exp_str="--experiment ${expname} --sub_experiment ${subexp}"
else
  exp_str="--experiment ${exparg}"
fi

if [[ $PYTHON == '' ]]; then
  python="python"
else
  python=$PYTHON
fi

$PYTHON batch_preprocess.py ${exp_str} ${GPU} > ${logfile} 2>${errfile}
