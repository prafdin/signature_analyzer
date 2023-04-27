import os.path

import yaml

config = {}
config_file_name = 'config.yaml'

if os.path.exists(config_file_name):
    with open(config_file_name, 'r') as file:
        config = yaml.safe_load(file)
