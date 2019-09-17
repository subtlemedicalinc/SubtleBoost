commit=${commit:=$(git rev-parse HEAD | cut -c1-6)}
job_id=$(cat experiments/$1/config.json | python3 -c "import sys, json; print(json.load(sys.stdin)['inference'])" | sha1sum | awk '{print $1}' | cut -c1-6)
out_folder=${commit}_${job_id}
logfile=$2/log_inference_${out_folder}.log

exparg=$1

if [[ $exparg == *"/"* ]]; then
  expname="$(cut -d'/' -f1 <<<"$exparg")"
  subexp="$(cut -d'/' -f2 <<<"$exparg")"
  exp_str="--experiment ${expname} --sub_experiment ${subexp}"
else
  exp_str="--experiment ${exparg}"
fi

python inference_process.py ${exp_str} --out_folder ${out_folder} > ${logfile} 2>${logfile}
