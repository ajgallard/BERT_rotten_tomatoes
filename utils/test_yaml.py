from yaml_func import yaml_multi_loader

content = yaml_multi_loader('.//config//config.yaml')
for i in range(len(content)):
    print(content[i])
