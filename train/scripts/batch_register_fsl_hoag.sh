#!/bin/bash

export dicom_data='/home/subtle/Data/SubtleGAD_Hoag'
export out_dir='/raid/jon/data_full_hoag/data'
export out_dir_plots='/raid/jon/data_full_hoag/plots'
export data_list='/raid/jon/data_full_hoag/data_list.txt'
export log_dir='/raid/jon/data_full_hoag/brain_extract/'
export par_jobs=1

mkdir -p ${out_dir}
mkdir -p ${out_dir_plots}

export max_files=1000

export brain_extract_bin=/home/subtle/jon/dev/ANTs/Scripts/antsBrainExtraction.sh
export template=/home/subtle/jon/dev/MICCAI2012-Multi-Atlas-Challenge-Data/T_template0.nii.gz
export mask=/home/subtle/jon/dev/MICCAI2012-Multi-Atlas-Challenge-Data/T_template0_BrainCerebellumProbabilityMask.nii.gz

export ANTSPATH=/home/subtle/jon/dev/ANTs/build/bin

function brain_extract() {

     typeset sub_dir="$@"

     sub_dir_no_spaces=$(echo "${sub_dir}" | sed 's/ //g' | sed 's/\///g')
     logfile=${log_dir}/${sub_dir_no_spaces}.out
     errfile=${log_dir}/${sub_dir_no_spaces}.err
     out_sub_dir=${out_dir}/${sub_dir_no_spaces}

     ls "${out_sub_dir}" | xargs -n1 -I {} echo ${brain_extract_bin} -d 3 -a ${out_sub_dir}/{} -e ${template} -m ${mask} -o ${out_sub_dir}/brain_{}
     ls "${out_sub_dir}" | xargs -n1 -I {} ${brain_extract_bin} -d 3 -a ${out_sub_dir}/{} -e ${template} -m ${mask} -o ${out_sub_dir}/brain_{} > ${logfile} 2>${errfile}
}

export -f brain_extract


mkdir -p ${log_dir}
cat $data_list | head -${max_files} | xargs -n1 -P ${par_jobs} -I{} bash -c "brain_extract {}"

