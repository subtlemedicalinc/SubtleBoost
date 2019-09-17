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

fcontent=$(python3 ./utils/print_config_json.py experiments/${expname}/config.json train)
fcontent=${fcontent}--${exparg}

job_id=$(echo ${fcontent} | sha1sum | awk '{print $1}' | cut -c1-6)
logfile=$2/log_train_${commit}_${job_id}.log

python train_process.py ${exp_str} ${gpu_str} > ${logfile} 2>${logfile}
