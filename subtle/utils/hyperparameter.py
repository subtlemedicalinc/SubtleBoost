import os
import json
import warnings

try:
    from test_tube import HyperOptArgumentParser
except:
    warnings.warn('Module test_tube not found - hyperparameter related functions cannot be used')

from . import experiment as utils_exp
from . import misc as utils_misc

def get_tunable_params(hypsearch_name, dirpath_hyp='./configs/hyperparam'):
    fpath_json = os.path.join(dirpath_hyp, '{}.json'.format(hypsearch_name))
    json_str = open(fpath_json, 'r').read()
    hyp_config = json.loads(json_str)

    return (
        hyp_config['tunable']['experiment'], hyp_config['tunable']['model']
    )

def get_hypsearch_params(hypsearch_name, dirpath_hyp='./configs/hyperparam'):
    fpath_json = os.path.join(dirpath_hyp, '{}.json'.format(hypsearch_name))
    json_str = open(fpath_json, 'r').read()
    hyp_config = json.loads(json_str)

    exp_splits = hyp_config['base_experiment'].split('/')
    experiment = exp_splits[0]

    if len(exp_splits) == 2:
        sub_experiment = exp_splits[1]
    else:
        sub_experiment = None

    default_config = utils_exp.get_config(experiment, sub_experiment, config_key='train')

    hyp_hash = utils_misc.get_timestamp_hash()
    hyp_log_dir = os.path.join(hyp_config['log_dir'], '{}_{}'.format(hypsearch_name, hyp_hash))
    default_config.config_dict['hyp_log_dir'] = hyp_log_dir

    if not os.path.exists(hyp_log_dir):
        os.makedirs(hyp_log_dir)

    tunable_config = hyp_config['tunable']['experiment']
    arch_tunable = {
        '__model_{}'.format(k): v
        for k, v in hyp_config['tunable']['model'].items()
    }
    tunable_config = {**tunable_config, **arch_tunable}

    hparser = HyperOptArgumentParser(strategy=hyp_config['strategy'])

    # add the non tunable params
    for key, val in default_config.config_dict.items():
        if isinstance(val, dict) or key in tunable_config:
            continue

        def_val = val if not key == 'experiment' else experiment
        hparser.add_argument('--{}'.format(key), default=def_val, type=type(def_val))

    # add the tunable params
    for key, val in tunable_config.items():
        opt_name = '--{}'.format(key)
        opt_default = default_config.config_dict.get(key)
        opt_type = type(opt_default)

        if val['type'] == 'range':
            hparser.opt_range(opt_name, type=opt_type, tunable=True, default=opt_default, low=val['low'], high=val['high'], nb_samples=hyp_config['trials'])
        elif val['type'] == 'list':
            hparser.opt_list(opt_name, type=opt_type, tunable=True, options=val['options'])
        else:
            raise ValueError('Tunable type "{}" not supported'.format(tunable['type']))

    hparams = hparser.parse_args()

    return hparams, hyp_config

def get_hyp_plot_list(hypsearch_name, dirpath_hyp='./configs/hyperparam'):
    fpath_json = os.path.join(dirpath_hyp, '{}.json'.format(hypsearch_name))
    json_str = open(fpath_json, 'r').read()
    hyp_config = json.loads(json_str)

    return hyp_config.get('plot')
