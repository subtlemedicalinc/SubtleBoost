#!/bin/bash

export dicom_data='/home/srivathsa/projects/studies/gad/tiantan'
export out_dir='/home/srivathsa/projects/studies/gad/tiantan_pp/data'
export out_dir_plots='/home/srivathsa/projects/studies/gad/tiantan_pp/plots'
export data_list='/home/srivathsa/projects/studies/gad/tiantan_pp/data_list.txt'
export log_dir='/home/srivathsa/projects/studies/gad/tiantan_pp/'
export par_jobs=1


mkdir -p ${out_dir}
mkdir -p ${out_dir_plots}

export max_files=1000

export preprocess_bin=/home/srivathsa/projects/SubtleGad/preprocess.py
export plot_grid_bin=/home/srivathsa/projects/SubtleGad/plot_grid.py

function preprocess() {

     typeset sub_dir="$@"

     sub_dir_no_spaces=$(echo "${sub_dir}" | sed 's/ //g' | sed 's/\///g')
     logfile=${log_dir}/${sub_dir_no_spaces}.out
     errfile=${log_dir}/${sub_dir_no_spaces}.err
     outfile=${out_dir}/${sub_dir_no_spaces}.h5
     outfile_png=${out_dir_plots}/${sub_dir_no_spaces}.png

     echo python $preprocess_bin --path_base "${dicom_data}/${sub_dir}" --verbose --output "${outfile}" --discard_start_percent 0 --discard_end_percent 0 --normalize --normalize_fun mean --transform_type "rigid" --scale_matching

     python $preprocess_bin --path_base "${dicom_data}/${sub_dir}" --verbose --output "${outfile}" --discard_start_percent 0 --discard_end_percent 0 --normalize --normalize_fun mean --transform_type "rigid" --scale_matching > ${logfile} 2>${errfile}

     python ${plot_grid_bin} --input $outfile --output ${outfile_png}

}

export -f preprocess

mkdir -p ${log_dir}
cat $data_list | head -${max_files} | xargs -n1 -P ${par_jobs} -I{} bash -c "preprocess {}"
