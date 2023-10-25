
import yaml
import argparse
import os
import json

parser = argparse.ArgumentParser(description='Change config')
parser.add_argument('path', type=str, help='path to SubtleMR')

if __name__ == "__main__":
    args = parser.parse_args()
    os.system(f"python3.10 {args.path}/subtle-app-utilities/subtle_python_packages/subtle/util/licensing.py 3000 SubtleMR 7989A8C0-A8E6-11E9-B934-238695B323F8 100 > {args.path}/a.yml")
    with open(f"{args.path}/a.yml", "r") as stream:
        a = (yaml.safe_load(stream))
    with open(f'{args.path}/dist/licenseMR.json', 'w') as fp:
        json.dump(a, fp)

    os.system(f"cp -r {args.path}/dist/licenseMR.json {args.path}/dist/SubtleMR/")
    os.system(f'chmod +x {args.path}/dist/SubtleMR/config.yml')

    with open(f'{args.path}/dist/SubtleMR/config.yml', 'r') as file:
        config_keys =yaml.safe_load(file)
    config_keys['jobs'][0]['exec_config'].update(series_desc_suffix = "")
    config_keys['series'][-1].update(reg_match = "")
    config_keys['series'][-1].update(reg_exclude= "")
        
    with open(f'{args.path}/dist/SubtleMR/configcopy.yml', 'w') as file:
        yaml.dump(config_keys, file)


