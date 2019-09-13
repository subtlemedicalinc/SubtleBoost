logfile=$2/preprocess.out
errfile=$2/preprocess.err
python batch_preprocess.py --experiment $1 > ${logfile} 2>${errfile}
