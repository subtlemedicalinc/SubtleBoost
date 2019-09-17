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
