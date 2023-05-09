import os
import traceback

import subtle.utils.experiment as utils_exp
import subtle.subtle_args as sargs

from plot_grid import plot_h5, plot_multi_contrast, save_video
from preprocess import execute_chain as preprocess_chain, preprocess_multi_contrast

if __name__ == '__main__':
    parser = sargs.get_parser()
    args = parser.parse_args()

    config = utils_exp.get_config(args.experiment, args.sub_experiment, config_key='preprocess')

    if config.data_batch:
        batch_splits = config.data_batch.split(',')
        batch_start = int(batch_splits[0])
        batch_end = int(batch_splits[1])
    else:
        batch_start = batch_end = None

    data_list = utils_exp.get_experiment_data(args.experiment, dataset='all', start=batch_start, end=batch_end)

    print('Processing the following cases...')
    print(data_list)

    if args.gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    for case_num in data_list:
        print('\n-------------\n')
        print('*****Processing {}*****\n'.format(case_num))

        config = utils_exp.get_config(args.experiment, args.sub_experiment, config_key='preprocess')
        config.path_base = os.path.join(config.dicom_data, case_num)
        config.out_file = os.path.join(config.out_dir, '{}.{}'.format(case_num, config.file_ext))

        if config.multi_contrast_mode:
            fext = '.{}'.format(config.file_ext)
            suffix = '_T2.{}'.format(config.file_ext) if 't2' in config.multi_contrast_kw else '_FLAIR.{}'.format(config.file_ext)
            config.out_file = config.out_file.replace(fext, suffix)

        try:
            outfile_png = os.path.join(config.out_dir_plots, '{}.png'.format(case_num))
            if config.multi_contrast_mode:
                preprocess_multi_contrast(config)
                plot_multi_contrast(input=config.out_file, output=outfile_png)
            else:
                preprocess_chain(config)
                plot_h5(input=config.out_file, output=outfile_png)

                if config.save_preprocess_video:
                    video_out = outfile_png.replace('png', 'mp4')
                    save_video(input=config.out_file, output=video_out)

                if config.fsl_mask:
                    outfile_png_mask = os.path.join(config.out_dir_plots, '{}_mask.png'.format(case_num))
                    plot_h5(input=config.out_file, output=outfile_png_mask, h5_key='data_mask')
        except Exception as err:
            print('PREPROCESSING ERROR in {}'.format(case_num))
            traceback.print_exc()
