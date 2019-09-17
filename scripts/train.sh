commit=${commit:=$(git rev-parse HEAD | cut -c1-6)}
exparg=$1

if [[ $exparg == *"/"* ]]; then
  expname="$(cut -d'/' -f1 <<<"$exparg")"
  subexp="$(cut -d'/' -f2 <<<"$exparg")"
  exp_str="--experiment ${expname} --sub_experiment ${subexp}"
else
  exp_str="--experiment ${exparg}"
fi

fcontent=$(cat experiments/${expname}/config.json | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin)['inference'], sort_keys=True))")
fcontent=${fcontent}--${exparg}

job_id=$(echo ${fcontent} | sha1sum | awk '{print $1}' | cut -c1-6)
logfile=$2/log_train_${commit}_${job_id}.log

python train_process.py ${exp_str} > ${logfile} 2>${logfile}
