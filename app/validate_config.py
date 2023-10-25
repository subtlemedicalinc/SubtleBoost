import sys
import os
from infer import SubtleSynthApp
from subtle.util.subtle_app import get_subtle_logger

LOGGER, _ = get_subtle_logger('ValidateConfig')

SCRIPT_DIR = os.path.dirname(os.path.dirname(__file__))
config_path = sys.argv[1]
manifest_info = os.path.join(SCRIPT_DIR, "manifest.json")
app = SubtleSynthApp(manifest_info=manifest_info)
manifest_exit_code, manifest_exit_detail = app._parse_manifest_info()
if manifest_exit_code != 0:
    LOGGER.info(f' {manifest_exit_detail}, {sys.stderr}')
    sys.exit(manifest_exit_code)
load_exit_code, load_exit_detail = app._load_config(config_path)
if load_exit_code != 0:
    LOGGER.info(f' {load_exit_detail} , {sys.stderr}')
    sys.exit(load_exit_code)
validate_exit_code, validate_exit_detail = app._validate_config()
if validate_exit_code != 0:
    LOGGER.info(f' {validate_exit_detail}, {sys.stderr}')
    sys.exit(validate_exit_code)
LOGGER.info("config is valid!")