import os

import subtle.utils.experiment as utils_exp
import subtle.subtle_args as sargs

from plot_grid import plot_h5
from train import train_process as train_execute

if __name__ == '__main__':
    parser = sargs.get_parser()
    args = parser.parse_args()

    config = utils_exp.get_config(args.experiment, args.sub_experiment, config_key='train')

    if config.save_all_weights:
        config.checkpoint_dir = os.path.join(config.checkpoint_dir, args.sub_experiment)

        if os.path.exists(config.checkpoint_dir):
            raise ValueError('Checkpoint directory {} already exists'.format(config.checkpoint_dir))
        os.makedirs(config.checkpoint_dir)

        config.checkpoint = os.path.join(
            config.checkpoint_dir, 'weights-{epoch:02d}-{val_loss:.2f}.checkpoint'
        )
    else:
        config.checkpoint = os.path.join(config.checkpoint_dir, '{}.checkpoint'.format(config.checkpoint_name))

    train_execute(config)
