commit=${commit:=$(git rev-parse HEAD | cut -c1-6)}
job_id=$(cat experiments/$1/config.json | python3 -c "import sys, json; print(json.load(sys.stdin)['inference'])" | sha1sum | awk '{print $1}' | cut -c1-6)
logfile=$2/log_train_${commit}_${job_id}.log
python train_process.py --experiment $1 > ${logfile} 2>${logfile}
