import os
import yaml

def load_parameters():
    params_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../params.yaml"))
    with open(params_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['train']


def main():
    params = load_parameters()
    print(params)
    print(f"Training for {params['epochs']} epochs with learning rate {params['learning_rate']}")


if __name__ == '__main__':
    main()
