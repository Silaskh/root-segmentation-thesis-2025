import yaml
import torch

#Loads a YAML configuration file and processes specific parameters
#Returns the configuration as a dictionary
def load_yaml_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    loss_params = config.get("loss", {}).get("params", {})
    if "weight" in loss_params and isinstance(loss_params["weight"], list):
        config["loss"]["params"]["weight"] = torch.tensor(loss_params["weight"], dtype=torch.float32)
    if "pos_weight" in loss_params and isinstance(loss_params["pos_weight"], (list, float, int)):
        config["loss"]["params"]["pos_weight"] = torch.tensor(loss_params["pos_weight"], dtype=torch.float32)

    return config
