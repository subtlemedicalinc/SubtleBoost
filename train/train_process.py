import os
import time

import subtle.utils.experiment as utils_exp
import subtle.subtle_args as sargs

from plot_grid import plot_h5
from train import train_process as train_execute

if __name__ == '__main__':
    parser = sargs.get_parser()
    args = parser.parse_args()

    config = utils_exp.get_config(args.experiment, args.sub_experiment, config_key='train')

    exp_name = args.experiment if args.sub_experiment is None else '{}_{}'.format(args.experiment, args.sub_experiment)
    tstr = str(time.time()).split('.')[0]

    config.checkpoint_dir = os.path.join(
        config.checkpoint_dir, '{}_{}'.format(exp_name, tstr)
    )

    if not args.resume_from_checkpoint:
        os.makedirs(config.checkpoint_dir)

    train_execute(config)
