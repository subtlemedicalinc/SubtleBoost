#!/bin/bash

export dicom_data='/home/subtle/Data/Stanford/lowcon'
export out_dir='/home/subtle/jon/dev/SubtleGad/data'
export par_jobs=1

function preprocess() {

     typeset sub_dir=$1

     echo python preprocess.py --path_base ${dicom_data}/${sub_dir} --verbose --output ${out_dir}/${sub_dir}.npy --discard_start_percent .1 --discard_end_percent .1 > logs_preprocess/log_${sub_dir}.out
     python preprocess.py --path_base ${dicom_data}/${sub_dir} --verbose --output ${out_dir}/${sub_dir}.npy --discard_start_percent .1 --discard_end_percent .1 > logs_preprocess/log_${sub_dir}.out 2>&1

}

export -f preprocess


mkdir -p logs_preprocess
ls $dicom_data | xargs -n1 -P ${par_jobs} -I{} bash -c "preprocess {}"

