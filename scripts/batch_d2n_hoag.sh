#!/bin/bash

export dicom_data='/home/subtle/Data/SubtleGAD_Hoag'
export out_dir='/raid/jon/data_full_hoag/data'
export out_dir_plots='/raid/jon/data_full_hoag/plots'
export data_list='/raid/jon/data_full_hoag/data_list.txt'
export log_dir='/raid/jon/data_full_hoag/d2n/'
export par_jobs=1

mkdir -p ${out_dir}
mkdir -p ${out_dir_plots}

export max_files=1000

export dicom2nifti_bin=/home/subtle/tensorflow/bin/dicom2nifti

function d2n() {

     typeset sub_dir="$@"

     sub_dir_no_spaces=$(echo "${sub_dir}" | sed 's/ //g' | sed 's/\///g')
     logfile=${log_dir}/${sub_dir_no_spaces}.out
     errfile=${log_dir}/${sub_dir_no_spaces}.err

     mkdir -p ${out_dir}/${sub_dir_no_spaces}
     ls "${dicom_data}/${sub_dir}" | xargs -n1 -I {} echo $dicom2nifti_bin "${dicom_data}/${sub_dir}/{}" "${out_dir}/${sub_dir_no_spaces}"
     ls "${dicom_data}/${sub_dir}" | xargs -n1 -I {} $dicom2nifti_bin "${dicom_data}/${sub_dir}/{}" "${out_dir}/${sub_dir_no_spaces}" > ${logfile} 2>${errfile}

}

export -f d2n


mkdir -p ${log_dir}
cat $data_list | head -${max_files} | xargs -n1 -P ${par_jobs} -I{} bash -c "d2n {}"

