import os
import json
import re as regex

import subtle.subtle_args as sargs
import subtle.utils.misc as utils_misc
from scripts.utils.print_config_json import IGNORE_KEYS as OVERRIDE_CONFIG_KEYS

def get_config(exp_name, subexp_name=None, config_key='preprocess', dirpath_exp='./configs/experiments'):
    class _ExperimentConfig:
        def __init__(self, config_dict):
            self.config_dict = config_dict
            parser = sargs.get_parser()
            ns_vars = vars(parser.parse_args())

            self.config_dict = {**ns_vars, **self.config_dict}
            for key, val in self.config_dict.items():
                setattr(self, key, val)

            for or_key in OVERRIDE_CONFIG_KEYS:
                if ns_vars[or_key] is not None:
                    self.config_dict[or_key] = ns_vars[or_key]
                    setattr(self, or_key, ns_vars[or_key])

        def debug_print(self):
            print('ExperimentConfig...\n')
            print(', '.join(['{}:{}'.format(key, val) for key, val in self.config_dict.items()]))

        def __str__(self):
            return ', '.join(['{}:{}'.format(key, val) for key, val in self.config_dict.items()])

    fname = 'config.json'
    fpath_json = os.path.join(dirpath_exp, exp_name, fname)

    if not os.path.exists(fpath_json):
        raise ValueError("Given experiment name {}, is not valid".format(exp_name))

    json_str = open(fpath_json, 'r').read()

    all_config = json.loads(json_str)
    gen_config = {k:v for k, v in all_config.items() if not isinstance(v, dict)} # get the top level config
    config_dict = all_config[config_key]
    config_dict = {**config_dict, **gen_config}

    if subexp_name is not None:
        subexp_config = config_dict[subexp_name]
        del config_dict[subexp_name]
        config_dict = {**config_dict, **subexp_config}

    return _ExperimentConfig(config_dict)

def get_model_config(model_name, config_key='base', model_type='generators', dirpath_config='./configs/models'):
    fpath_json = os.path.join(dirpath_config, model_type, '{}.json'.format(model_name))
    if not os.path.exists(fpath_json):
        raise ValueError("Given model name {}, is not valid".format(model_name))

    json_str = open(fpath_json, 'r').read()
    all_config = json.loads(json_str)
    base_config = all_config['base']
    if config_key == 'base':
        return base_config

    sub_config = all_config[config_key]
    return utils_misc.dict_merge(base_config, sub_config)

def get_experiment_data(exp_name, dirpath_exp='./configs/experiments', dataset='all'):
    fpath_json = os.path.join(dirpath_exp, exp_name, 'data.json')
    json_str = open(fpath_json, 'r').read()
    data_dict = json.loads(json_str)

    train_data = data_dict['train']
    test_data = data_dict['test']

    val_data = data_dict.get('validation')
    val_data = [] if val_data is None else val_data

    plot_data = data_dict.get('plot')

    data = None

    if dataset == 'train':
        data = train_data
    elif dataset == 'test':
        data = test_data
    elif dataset == 'val':
        data = val_data
    elif dataset == 'plot':
        data = plot_data
    else:
        data = train_data + test_data + val_data

    return data

def match_layer_to_config(config_dict, layer_name):
    layer_config = {}
    for key, val in config_dict.items():
        if not isinstance(val, dict): continue
        k = '^{}$'.format(key.replace('_', '-').replace('*', '(.*)'))
        ln = layer_name.replace('_', '-')
        if regex.match(k, ln):
            layer_config = {**layer_config, **val}
    return layer_config

def get_layer_config(config_dict, param_name, layer_name=''):
    try:
        layer_config = match_layer_to_config(config_dict, layer_name)
        if param_name in layer_config:
            return layer_config[param_name]
        return config_dict['all'][param_name]
    except KeyError:
        raise ValueError('Parameter "{}" not defined for layer - "{}"'.format(param_name, layer_name))
