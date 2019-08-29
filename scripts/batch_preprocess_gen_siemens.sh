#!/bin/bash

export dicom_data='/home/srivathsa/projects/studies/gad/gen_siemens/data'
export out_dir='/home/srivathsa/projects/studies/gad/gen_siemens/preprocess/data'
export out_dir_plots='/home/srivathsa/projects/studies/gad/gen_siemens/preprocess/plots'
export data_list='/home/srivathsa/projects/SubtleGad/data_lists/data_full_gen_siemens.txt'
export log_dir='/home/srivathsa/projects/studies/gad/gen_siemens/preprocess'
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
     outfile_full_png=${out_dir_plots}/${sub_dir_no_spaces}_full.png

     echo python $preprocess_bin --path_base "${dicom_data}/${sub_dir}" --verbose --output "${outfile}" --discard_start_percent 0 --discard_end_percent 0 --normalize --normalize_fun mean --joint_normalize --transform_type "affine" --skip_scale_im --mask_threshold .1 --scale_matching --noise_mask_area --resample_isotropic --scale_dicom_tags --fsl_mask --pad_for_size 256

     python $preprocess_bin --path_base "${dicom_data}/${sub_dir}" --verbose --output "${outfile}" --discard_start_percent 0 --discard_end_percent 0 --normalize --normalize_fun mean --joint_normalize --transform_type "affine" --skip_scale_im --mask_threshold .1 --scale_matching --resample_isotropic --scale_dicom_tags --fsl_mask --pad_for_size 256 > ${logfile} 2>${errfile}

     python ${plot_grid_bin} --input $outfile --output ${outfile_png} --h5_key data_mask

     python ${plot_grid_bin} --input $outfile --output ${outfile_full_png} --h5_key data
}

export -f preprocess


mkdir -p ${log_dir}
cat $data_list | head -${max_files} | xargs -n1 -P ${par_jobs} -I{} bash -c "preprocess {}"
