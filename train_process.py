import os

import subtle.utils.experiment as exp_utils
import subtle.subtle_args as sargs

from plot_grid import plot_h5
from train import train_process as train_execute

if __name__ == '__main__':
    parser = sargs.get_parser()
    args = parser.parse_args()

    config = exp_utils.get_config(args.experiment, args.sub_experiment, config_key='train')

    config.checkpoint = os.path.join(config.checkpoint_dir, '{}.checkpoint'.format(config.checkpoint_name))

    train_execute(config)
