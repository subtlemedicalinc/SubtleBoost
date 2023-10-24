
import yaml
import argparse
import os

parser = argparse.ArgumentParser(description='Change config')
parser.add_argument('path', type=str, help='path to SubtleMR')

if __name__ == "__main__":
    args = parser.parse_args()

    os.system(f'chmod +x {args.path}/config.yml')
    with open(f'{args.path}/config.yml', 'r') as file:
        config_keys =yaml.safe_load(file)
    config_keys['jobs'][0]['exec_config'].update(series_desc_suffix = "")
    config_keys['series'][-1].update(reg_match = "")
    config_keys['series'][-1].update(reg_exclude= "")
        
    with open(f'{args.path}/configcopy.yml', 'w') as file:
        yaml.dump(config_keys, file)


