from copy import deepcopy

import yaml
from SED_model2 import *

def get_models(configs, train_cfg, multigpu):
    net = CRNN(**configs["CRNN"])
    # ema network
    ema_net = deepcopy(net)
    for param in ema_net.parameters():
        param.detach_()

    if multigpu and (train_cfg["n_gpu"] > 1):
        net = nn.DataParallel(net)
        ema_net = nn.DataParallel(ema_net)

    net = net.to(train_cfg["device"])
    ema_net = ema_net.to(train_cfg["device"])
    return net, ema_net

def get_configs(config_dir, server_config_dir=r"C:\Users\issac\Documents\ML\Yolov8\Code\SED\SED_config_server.yaml"):
    #get hyperparameters from yaml
    with open(config_dir, "r") as f:
        configs = yaml.safe_load(f)
    with open(server_config_dir, "r") as f:
        server_cfg = yaml.safe_load(f)

    train_cfg = configs["training"]
    feature_cfg = configs["feature"]
    train_cfg["batch_sizes"] = server_cfg["batch_size"]
    train_cfg["net_pooling"] = feature_cfg["net_subsample"]
    return configs, server_cfg, train_cfg, feature_cfg