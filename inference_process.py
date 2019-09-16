import os
import traceback

import subtle.subtle_io as suio
import subtle.subtle_args as sargs

from plot_grid import plot_h5
from inference import inference_process as run_inference

if __name__ == '__main__':
    parser = sargs.parser()
    args = parser.parse_args()

    data_list = suio.get_experiment_data(args.experiment, dataset='test')

    if not os.path.exists()

    for case_num in data_list:
        print('\n-------------\n')
        print('*****Running inference for {}*****\n'.format(case_num))

        config = suio.get_config(args.experiment, config_key='inference')
        config.checkpoint_file = '{}/{}'.format(config.checkpoint_dir, config.checkpoint)

        config.data_preprocess = '{}/{}.h5'.format(config.data_dir, case_num)
        config.path_out = '{}/{}/{}/{}_SubtleGad'.format(config.data_raw, config.out_folder, case_num, case_num)
        config.path_base = '{}/{}'.format(config.data_raw, case_num)

        metrics_dir = '{}/metrics/{}'.format(config.stats_base, args.description)
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)

        config.stats_file = '{}/{}.h5'.format(metrics_dir, case_num)

        try:
            run_inference(config)
        except Exception as err:
            print('INFERENCE ERROR in {}'.format(case_num))
            traceback.print_exc()
