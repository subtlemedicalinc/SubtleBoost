<<<<<<< HEAD
import os
import traceback
=======
import sys
import os
import traceback
import configargparse as argparse
>>>>>>> Preprocess pipeline using experiment configs

import subtle.subtle_io as suio
import subtle.subtle_args as sargs

from plot_grid import plot_h5
from preprocess import execute_chain as preprocess_chain

if __name__ == '__main__':
    parser = sargs.parser()
    args = parser.parse_args()

    data_list = suio.get_experiment_data(args.experiment, dataset='all')

    for case_num in data_list:
        print('\n-------------\n')
        print('*****Processing {}*****\n'.format(case_num))

        config = suio.get_config(args.experiment, args.sub_experiment, config_key='preprocess')
        config.path_base = os.path.join(config.dicom_data, case_num)
        config.out_file = os.path.join(config.out_dir, '{}.h5'.format(case_num))

        try:
            preprocess_chain(config)
        except Exception as err:
            print('PREPROCESSING ERROR in {}'.format(case_num))
            traceback.print_exc()
        outfile_png = os.path.join(config.out_dir_plots, '{}.png'.format(case_num))
        plot_h5(input=config.out_file, output=outfile_png)

        if config.fsl_mask:
            outfile_png_mask = os.path.join(config.out_dir_plots, '{}_mask.png'.format(case_num))
            plot_h5(input=config.out_file, output=outfile_png_mask, h5_key='data_mask')
