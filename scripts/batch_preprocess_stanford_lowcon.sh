#!/bin/bash

export dicom_data='/home/subtle/Data/Stanford/lowcon'
export out_dir='/raid/jon/data_full_stanford/data'
export out_dir_plots='/raid/jon/data_full_stanford/plots'
export data_list='/raid/jon/data_full_stanford/data_list.txt'
export log_dir='/raid/jon/data_full_stanford/preprocess/'
export par_jobs=1


mkdir -p ${out_dir}
mkdir -p ${out_dir_plots}

export max_files=1000

export preprocess_bin=/home/subtle/jon/dev/SubtleGad/preprocess.py
export plot_grid_bin=/home/subtle/jon/dev/SubtleGad/plot_grid.py

function preprocess() {

     typeset sub_dir="$@"

     sub_dir_no_spaces=$(echo "${sub_dir}" | sed 's/ //g' | sed 's/\///g')
     logfile=${log_dir}/${sub_dir_no_spaces}.out
     errfile=${log_dir}/${sub_dir_no_spaces}.err
     outfile=${out_dir}/${sub_dir_no_spaces}.h5
     outfile_png=${out_dir_plots}/${sub_dir_no_spaces}.png

     python2 /home/ubuntu/jon/dev/SubtleGad/preprocess.py --path_base ${dicom_data}/${sub_dir} --verbose --output ${out_dir}/${sub_dir}.npy --discard_start_percent .1 --discard_end_percent .1 --normalize --normalize_fun mean > logs_preprocess/log_${sub_dir}.out 2>&1

     echo python $preprocess_bin --path_base "${dicom_data}/${sub_dir}" --verbose --output "${outfile}" --discard_start_percent 0 --discard_end_percent 0 --normalize --normalize_fun mean --transform_type "affine" --scale_matching 

     python $preprocess_bin --path_base "${dicom_data}/${sub_dir}" --verbose --output "${outfile}" --discard_start_percent 0 --discard_end_percent 0 --normalize --normalize_fun mean --transform_type "affine" --scale_matching > ${logfile} 2>${errfile}

     python ${plot_grid_bin} --input $outfile --output ${outfile_png}

}

export -f preprocess

mkdir -p ${log_dir}
cat $data_list | head -${max_files} | xargs -n1 -P ${par_jobs} -I{} bash -c "preprocess {}"
