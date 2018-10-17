#!/bin/bash

export dicom_data='/home/subtle/Data/SubtleGAD_Hoag'
export out_dir='/raid/jon/data_full_hoag/data'
export out_dir_plots='/raid/jon/data_full_hoag/plots'
export data_list='/raid/jon/data_full_hoag/data_list.txt'
export par_jobs=1
export log_dir='/raid/jon/data_full_hoag/preprocess/'

mkdir -p ${out_dir}
mkdir -p ${out_dir_plots}

export max_files=1000

export preprocess_bin=/home/subtle/jon/dev/SubtleGad/preprocess.py
export plot_grid_bin=/home/subtle/jon/dev/SubtleGad/plot_grid.py

function preprocess() {

     typeset sub_dir="$@"

     sub_dir_no_spaces=$(echo "${sub_dir}" | sed 's/ //g' | sed 's/\///g')
     logfile=${log_dir}/${sub_dir_no_spaces}.out
     outfile=${out_dir}/${sub_dir_no_spaces}.npy
     outfile_png=${out_dir_plots}/${sub_dir_no_spaces}.png

     echo python $preprocess_bin --path_base "${dicom_data}/${sub_dir}" --verbose --output "${outfile}" --discard_start_percent 0 --discard_end_percent 0 --normalize --normalize_fun mean --transform_type "affine" --skip_scale_im --joint_normalize
     python $preprocess_bin --path_base "${dicom_data}/${sub_dir}" --verbose --output "${outfile}" --discard_start_percent 0 --discard_end_percent 0 --normalize --normalize_fun mean --transform_type "affine" --skip_scale_im --joint_normalize > ${logfile} 2>&1


     python ${plot_grid_bin} --input $outfile --output ${outfile_png}

}

export -f preprocess


mkdir -p ${log_dir}
cat $data_list | head -${max_files} | xargs -n1 -P ${par_jobs} -I{} bash -c "preprocess {}"

