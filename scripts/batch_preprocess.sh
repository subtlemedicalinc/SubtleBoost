#!/bin/bash

export dicom_data='/data/test_cloud_inference/lowcon'
export out_dir='/home/ubuntu/jon/dev/data_full/data'
export out_dir_plots='/home/ubuntu/jon/dev/data_full/data/plots'
export par_jobs=1

mkdir -p ${out_dir}
mkdir -p ${out_dir_plots}

export max_files=1000

function preprocess() {

     typeset sub_dir=$1

     echo python2 /home/ubuntu/jon/dev/SubtleGad/preprocess.py --path_base ${dicom_data}/${sub_dir} --verbose --output ${out_dir}/${sub_dir}.npy --discard_start_percent .1 --discard_end_percent .1 --normalize --normalize_fun mean > logs_preprocess/log_${sub_dir}.out
     python2 /home/ubuntu/jon/dev/SubtleGad/preprocess.py --path_base ${dicom_data}/${sub_dir} --verbose --output ${out_dir}/${sub_dir}.npy --discard_start_percent .1 --discard_end_percent .1 --normalize --normalize_fun mean > logs_preprocess/log_${sub_dir}.out 2>&1

     python /home/ubuntu/jon/dev/SubtleGad/plot_grid.py --input ${out_dir}/${sub_dir}.npy --output ${out_dir_plots}/${sub_dir}.png

}

export -f preprocess


mkdir -p logs_preprocess
ls $dicom_data | grep "Patient" | head -${max_files} | xargs -n1 -P ${par_jobs} -I{} bash -c "preprocess {}"

