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

if test -f experiments/${expname}/config.json; then
  echo ""
else
  echo "Invalid experiment name - ${expname}"
  exit 1
fi

fcontent=$(python3 ./utils/print_config_json.py experiments/${expname}/config.json inference)
fcontent=${fcontent}--${exparg}

job_id=$(echo ${fcontent} | sha1sum | awk '{print $1}' | cut -c1-6)
out_folder=${commit}_${job_id}
logfile=$2/log_inference_${out_folder}.log

python inference_process.py ${exp_str} ${gpu_str} --out_folder ${out_folder} > ${logfile} 2>${logfile}
