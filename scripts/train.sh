timestamp=$(date "+%s")
logfile=$2/log_$1_${timestamp}.log
python train_process.py --experiment $1 > ${logfile} 2>${logfile}
