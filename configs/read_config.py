import yaml

with open("./configs/config.yaml", "r") as f:
    content = yaml.safe_load(f)


def get_config(name):
    return content[name]
