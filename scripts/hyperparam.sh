#!/bin/bash

function show_usage() {
	echo "Usage: $0 hypsearch_name log_dir"
}

function show_help() {
	echo ""
	echo "SubtleGad hyperparameter search convenience script."
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

exparg=$1

if test -f configs/hyperparam/${exparg}.json; then
  echo ""
else
  echo "Invalid hypsearch config name - ${exparg}"
  exit 1
fi

logfile=$2/log_hypsearch_${exparg}.log
python hyperparam.py --hypsearch_name ${exparg} > ${logfile} 2>${logfile}
