import argparse
import time

# import csv
import yaml
import os
from superpoint.utils.logging import *
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from tensorboardX import SummaryWriter

# from utils.utils import tensor2array, save_checkpoint, load_checkpoint, save_path_formatter
from superpoint.utils.utils import (
    tensor2array,
    save_checkpoint,
    load_checkpoint,
    save_path_formatter,
)
from deepFEPE.settings import EXPER_PATH

from imgaug import augmenters as iaa
import imgaug as ia


def get_save_path(output_dir):
    save_path = Path(output_dir)
    save_path = save_path / "checkpoints"
    logging.info("=> will save everything to {}".format(save_path))
    os.makedirs(save_path, exist_ok=True)
    return save_path


def worker_init_fn(worker_id, reproduce=False):
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed.
    Please try to be consistent.

    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed

    """
    if reproduce:
        torch.manual_seed(0)
        np.random.seed(0)
    else:
        base_seed = torch.IntTensor(1).random_().item()
        np.random.seed(base_seed + worker_id)

    print(f"base_seed = {base_seed}, workder_id = {worker_id}")
    # print(worker_id, base_seed)


def dataLoader(
    config, dataset="syn", warp_input=False, train=True, val="val", val_shuffle=True
):
    import torchvision.transforms as transforms

    data_transforms = {
        "train": transforms.Compose([transforms.ToTensor(),]),
        "val": transforms.Compose([transforms.ToTensor(),]),
    }
    if dataset == "kitti_odo_corr":
        from deepFEPE.datasets.kitti_odo_corr import KittiCorrOdo as Dataset
    else:
        Dataset = get_module("datasets", dataset)
        # from datasets.coco import Coco as Dataset
    val_set = Dataset(transform=data_transforms["train"], task=val, **config,)
    logging.info(
        "Creating val loader with %d workers..." % config["training"]["workers_val"]
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=config["data"]["batch_size"],
        shuffle=val_shuffle,
        pin_memory=True,
        num_workers=config["training"]["workers_val"],
        worker_init_fn=None if config["training"]["reproduce"] else worker_init_fn,
    )
    if train:
        train_set = Dataset(transform=data_transforms["train"], task="train", **config,)
        logging.info(
            "Creating train loader with %d workers..."
            % config["training"]["workers_train"]
        )
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config["data"]["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=config["training"]["workers_train"],
            worker_init_fn=None if config["training"]["reproduce"] else worker_init_fn,
        )
    else:
        logging.warning(f"not training. use val as dummy training set")
        train_set, train_loader = val_set, val_loader
    # val_set, val_loader = None, None
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_set": train_set,
        "val_set": val_set,
    }



# def modelLoader(model, params):
def modelLoader(model, **params):
    # create model
    logging.info(f"=> creating model: {model}")
    if model == "GoodCorresNet_layers_deepF":
        # if params['if_learn_offsets']:
        #     from models.DeepFNetOffset import Norm8PointNetOffset
        #     net = Norm8PointNetOffset(**params)
        # else:
        if params["if_sample_loss"]: # default as false
            from models.DeepFNetSampleLoss import Norm8PointNet as model
        else:
            from models.DeepFNet import DeepFNet as model
        net = model(**params)
    # elif model == "GoodCorresNet_layers_deepF_SP":
    #     from models.DeepFNet_SP import Norm8PointNet_SP

    #     net = Norm8PointNet_SP(**params)
    # elif model == "GoodCorresNet_layers_deepF_singleSample":
    #     from models.DeepFNet_singleSample import Norm8PointNetSingleSample

    #     net = Norm8PointNetSingleSample(**params)
    #     # from models.DeepFNetDes import Norm8PointNetDes
    #     # net = Norm8PointNetDes(**params)
    else:
        logging.error(f"Model {model} not defined specifically!")
        # create model
        logging.info("=> creating model: %s", model)
        net = get_model(model)
        net = net(**params)

    return net


def get_module(path, name):
    import importlib

    if path == "":
        mod = importlib.import_module(name)
    else:
        mod = importlib.import_module("{}.{}".format(path, name))
    return getattr(mod, name)


def get_model(name):
    mod = __import__("models.{}".format(name), fromlist=[""])
    return getattr(mod, name)


# def modelLoader(model='SuperPointNet', **options):
#     # create model
#     logging.info("=> creating model: %s", model)
#     net = get_model(model)
#     net = net(**options)
#     return net


# mode: 'full' means the formats include the optimizer and epoch
# full_path: if not full path, we need to go through another helper function
### deprecated ###
def pretrainedLoader(net, optimizer, epoch, path, mode="full", full_path=False):
    # load checkpoint
    print("This function is deprecated!!! Please use pretrainedLoader_net!!!")
    if full_path == True:
        checkpoint = torch.load(path)
    else:
        checkpoint = load_checkpoint(path)
    # apply checkpoint
    if mode == "full":
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        #         epoch = checkpoint['epoch']
        epoch = checkpoint["n_iter"]
    #         epoch = 0
    else:
        net.load_state_dict(checkpoint)
        # net.load_state_dict(torch.load(path,map_location=lambda storage, loc: storage))
    return net, optimizer, epoch


def pretrainedLoader_net(net, epoch, path, mode="full", full_path=False):
    # load checkpoint
    if full_path == True:
        checkpoint = torch.load(path)
    else:
        checkpoint = load_checkpoint(path)
    # apply checkpoint
    # print(checkpoint)
    assert mode == "full"
    net.load_state_dict(checkpoint["model_state_dict"])
    n_iter = checkpoint["n_iter"]
    if "n_iter_val" in checkpoint:
        n_iter_val = checkpoint["n_iter_val"]
    else:
        n_iter_val = 0
    logging.info(">>>>> Loaded %s" % path)
    return net, n_iter, n_iter_val


def pretrainedLoader_opt(optimizer, epoch, path, mode="full", full_path=False):
    # load checkpoint
    if full_path == True:
        checkpoint = torch.load(path)
    else:
        checkpoint = load_checkpoint(path)
    # apply checkpoint
    assert mode == "full"
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    n_iter = checkpoint["n_iter"]
    if "n_iter_val" in checkpoint:
        n_iter_val = checkpoint["n_iter_val"]
    else:
        n_iter_val = 0
    return optimizer, n_iter, n_iter_val
