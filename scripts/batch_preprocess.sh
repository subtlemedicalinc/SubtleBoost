#!/bin/bash

function show_usage() {
	echo "Usage: $0 experiment_name log_dir"
}

function show_help() {
	echo ""
	echo "SubtleGad batch preprocessing convenience script."
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

python batch_preprocess.py ${exp_str} > ${logfile} 2>${errfile}
