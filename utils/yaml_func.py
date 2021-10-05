import yaml

def yaml_loader(filepath):
    """Loads a yaml file"""
    with open(filepath, "r") as file_descriptor:
        data = yaml.load(file_descriptor, yaml.safe_load)
    return data

def yaml_multi_loader(filepath):
    """Loads a yaml file"""
    with open(filepath, "r") as file_descriptor:
        data = [k for k in yaml.safe_load_all(file_descriptor)]
    return data

def yaml_dump(filepath, data):
    """Dumps data to a yaml file"""
    with open(filepath, "w") as file_descriptor:
        yaml.dump(data, file_descriptor)

if __name__ == "main":
    yaml_loader()
    yaml_multi_loader()
    yaml_dump()
