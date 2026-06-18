import yaml 

with open('./config.yaml', 'r') as f:
    content = yaml.safe_load(f)


print("Model:", content['model_name'])
print("Learning rate:", content['learning_rate'])
print("Epochs:", content['epochs'])


