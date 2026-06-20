import os

import yaml

config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(config_path, "r") as f:
    content = yaml.safe_load(f)


def get_config(name):
    return content[name]
