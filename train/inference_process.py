import os
import traceback

import subtle.utils.experiment as utils_exp
import subtle.subtle_args as sargs

from plot_grid import plot_h5
from inference import inference_process as run_inference
from subtle.dnn.helpers import clear_keras_memory

if __name__ == '__main__':
    parser = sargs.get_parser()
    args = parser.parse_args()

    data_list = utils_exp.get_experiment_data(args.experiment, dataset='test')

    for case_num in data_list:
        print('\n-------------\n')
        print('*****Running inference for {}*****\n'.format(case_num))
        config = utils_exp.get_config(args.experiment, args.sub_experiment, config_key='inference')
        config.checkpoint_file = '{}/{}'.format(config.checkpoint_dir, config.checkpoint)

        path_base = '{}/{}'.format(config.data_dir, case_num)
        if config.dicom_inference:
            config.path_base = path_base
        else:
            config.data_preprocess = '{}.{}'.format(path_base, config.file_ext)

        config.path_out = '{}/{}/{}/{}_SubtleGad'.format(config.data_raw, config.out_folder, case_num, case_num)
        config.path_base = '{}/{}'.format(config.data_raw, case_num)

        metrics_dir = '{}/metrics/{}'.format(config.stats_base, config.description)
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)

        config.stats_file = '{}/{}.h5'.format(metrics_dir, case_num)

        clear_keras_memory()
        run_inference(config)
