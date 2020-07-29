""" 
# main training script
Reviewed and tested by You-Yi on 07/07/2020.

Edition history:
    Organized and documented by You-Yi on 05/18/2020.
    Increments upon train_good_corr_3_vals_goodF.py on 09/29/2019 by Rui. Reorganized as baseline deep_F.
    
Authors:
    You-Yi Jau, Rui Zhu
"""

import argparse
import yaml
import os, sys
from path import Path
import copy
import numpy as np
import cv2
import time
import scipy
from tqdm import tqdm
from pebble import ProcessPool
import multiprocessing as mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## torch
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from tensorboardX import SummaryWriter

# our modules
from settings import EXPER_PATH

from utils.loader import (
    dataLoader,
    modelLoader,
    pretrainedLoader_net,
    pretrainedLoader_opt,
)
from train_good_utils import (
    get_all_loss,
    val_rt,
    get_all_loss_DeepF,
    write_metrics_summary,
    get_Rt_loss,
    mean_list,
    get_matches_from_SP,
)

import dsac_tools.utils_F as utils_F  # If cannot find: export KITTI_UTILS_PATH='/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils'
import dsac_tools.utils_opencv as utils_opencv  # If cannot find: export KITTI_UTILS_PATH='/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils'
import dsac_tools.utils_vis as utils_vis  # If cannot find: export KITTI_UTILS_PATH='/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils'
import dsac_tools.utils_misc as utils_misc  # If cannot find: export KITTI_UTILS_PATH='/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils'
import dsac_tools.utils_geo as utils_geo  # If cannot find: export KITTI_UTILS_PATH='/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils'

from Train_model_pipeline import init_dict_of_lists
from Train_model_pipeline import Train_model_pipeline


# functions from superpoint
from superpoint.utils.logging import *
from superpoint.utils.print_tool import print_dict_attr
from superpoint.models.SuperPointNet_gauss2 import SuperPointNet_gauss2
from superpoint.models.model_utils import SuperPointNet_process
from superpoint.models.model_wrap import PointTracker
from superpoint.utils.utils import (
    tensor2array,
    save_checkpoint,
    load_checkpoint,
    save_path_formatter,
    getWriterPath,
    saveLoss,
    toNumpy,
)


# parameters
np.set_printoptions(precision=4, suppress=True)
ratio_CPU = 0.2 # 0.4
default_number_of_process = int(ratio_CPU * mp.cpu_count())
# default_number_of_process = 1


def eval_good(config, output_dir, args):
    """
    an abstract layer to differ eval and training results.
    """
    train_good(config, output_dir, args)
    pass


def train_good(config, output_dir, args):
    """
    # training script, controlled by config file and args
    # work with Train_model_pipeline.py
    params:
        config: config file path, contain the settings
        output_dir: the path for results
        args: some setting

    """
    # config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("train on device: %s", device)
    with open(os.path.join(output_dir, "config.yml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    writer = SummaryWriter(
        getWriterPath(task=args.command, exper_name=args.exper_name, date=True)
    )

    logging.info(f"config: {config}")

    ## reproducibility
    if config["training"]["reproduce"]:
        logging.info("reproduce = True")
        torch.manual_seed(0)
        np.random.seed(0)
        print(f"test random # : np({np.random.rand(1)}), torch({torch.rand(1)})")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ## save paths
    save_path = Path(output_dir)
    save_path = save_path / "checkpoints"
    logging.info("+++[Train]+++ will save everything to {}".format(save_path))
    os.makedirs(save_path, exist_ok=True)

    # data loading
    assert (
        config["data"]["sequence_length"] == 2
    ), "Sorry dude, we are only supporting two-frame setting for now."
    # assert (config['data']['read_what']['with_X'] and config['model']['batch_size']==1) or (not config['data']['read_what']['with_X']), 'We are not suppoting batching lidar Xs with batch_size>1 yet!'
    config["data"]["read_what"]["with_quality"] = config["model"]["if_quality"]

    val = "test" if args.test else "val"  # use test or val data
    val_shuffle = False if args.test else True  # not sorting when doing testing
    train = False if args.test else True
    data = dataLoader(
        config,
        dataset=config["data"]["dataset"],
        train=train,
        warp_input=True,
        val=val,
        val_shuffle=val_shuffle,
    )
    train_loader, val_loader = data["train_loader"], data["val_loader"]
    logging.info(
        "+++[Dataset]+++ train split size %d in %d batches, val split size %d in %d batches"
        % (
            len(train_loader) * config["data"]["batch_size"],
            len(train_loader),
            len(val_loader) * config["data"]["batch_size"],
            len(val_loader),
        )
    )

    # model loading
    # model_params = {'depth': config['model']['depth'], 'clamp_at':config['model']['clamp_at'], 'in_channels': 4, 'num_seg_classes': 1, 'with_transform': False, 'with_instance_norm': True}
    if config["model"]["if_SP"]:
        config["model"]["quality_size"] = 1
    img_zoom_xy = (
        config["data"]["preprocessing"]["resize"][1]
        / config["data"]["image"]["size"][1],
        config["data"]["preprocessing"]["resize"][0]
        / config["data"]["image"]["size"][0],
    )
    model_params = {
        "depth": config["model"]["depth"],
        "img_zoom_xy": img_zoom_xy,
        "image_size": config["data"]["image"]["size"],
        "quality_size": config["model"]["quality_size"],
        "if_quality": config["model"]["if_quality"],
        "if_img_des_to_pointnet": config["model"]["if_img_des_to_pointnet"],
        "if_goodCorresArch": config["model"]["if_goodCorresArch"],
        "if_img_feat": config["model"]["if_img_feat"],
        "if_cpu_svd": config["model"]["if_cpu_svd"],
        "if_learn_offsets": config["model"]["if_learn_offsets"],
        "if_tri_depth": config["model"]["if_tri_depth"],
        "if_sample_loss": config["model"]["if_sample_loss"],
    }

    ## load model and weights - deep fundametal network (deepF)
    net = modelLoader(config["model"]["name"], **model_params)
    print(f"deepF net: {net}")
    n_iter = 0
    n_iter_val = 0 + n_iter

    ## load model and weights - superpoint (sp)
    if config["model"]["if_SP"]:
        SP_params = {
            "out_num_points": 2000,  ### no use
            "patch_size": 5,
            "device": device,
            "nms_dist": 4,
            "conf_thresh": 0.015,
            "nn_thresh": 0.7,
        }
        params = config["training"].get("SP_params", None)
        if params is not None:
            SP_params.update(params)
        else:
            logging.warning(f"use default Superpoint Parameters")
        # for e in list(params):
        #     if e != 'device':
        #         params[e] = float(params[e])
        logging.info(f"SP_params: {SP_params}")

        # checkpoint_path_SP = "logs/superpoint_coco_heat2_0/checkpoints/superPointNet_170000_checkpoint.pth.tar"
        checkpoint_path_SP = config["training"]["pretrained_SP"]
        checkpoint_mode_SP = "" if checkpoint_path_SP[-3:] == "pth" else "full"
        # helping modules
        SP_processer = SuperPointNet_process(**SP_params)
        SP_tracker = PointTracker(max_length=2, nn_thresh=params["nn_thresh"])

        # load the network
        net_SP = SuperPointNet_gauss2()
        n_iter_SP = 0
        n_iter_val_SP = 0 + n_iter_SP

        ## load pretrained and create optimizer
        net_SP, optimizer_SP, n_iter_SP, n_iter_val_SP = prepare_model(
            config, net_SP, device, n_iter_SP, n_iter_val_SP, net_postfix="_SP"
        )
        # net_SP = nn.DataParallel(net_SP)  # AttributeError: 'DataParallel' object has no attribute 'process_output'
        if config["training"].get("train_SP", True):
            logging.info("+++[Train]+++  training superpoint")
        else:
            logging.info("+++[Train]+++  superpoint is used but not trained")

    ## load pretrained and create optimizer
    net, optimizer, n_iter, n_iter_val = prepare_model(
        config, net, device, n_iter, n_iter_val, net_postfix=""
    )
    if config["training"].get("train", True):
        logging.info("+++[Train]+++  training deepF model")
    else:
        logging.info("+++[Train]+++  deepF model is used but not trained")

    epoch = n_iter // len(train_loader)

    # set up train_agent
    train_agent = Train_model_pipeline(
        config, save_path=save_path, args=args, device=device
    )
    train_agent.writer = writer

    # feed the data into the agent
    train_agent.train_loader = train_loader
    train_agent.val_loader = val_loader
    train_agent.set_params(n_iter=n_iter, epoch=epoch, n_iter_val=n_iter_val)

    if not config["model"]["if_SP"]:
        net_SP = None
        optimizer_SP = None
        SP_processer = None
        SP_tracker = None
    train_agent.set_nets(net, net_SP=net_SP)
    train_agent.set_optimizers(optimizer, optimizer_SP)
    train_agent.set_SP_helpers(SP_processer, SP_tracker)

    while True:
        # Train for one epoch; val occasionally
        epoch_loss, _, n_iter, n_iter_val = train_agent.train_epoch(train=True)
        save_file = save_path / "training.txt"
        # saveLoss(save_file, n_iter, epoch_loss)
        if n_iter > config["training"]["train_iter"]:
            break

    print("Finished Training")


# def prepare_model(config, net, device, n_iter, n_iter_val, net_postfix=""):
def prepare_model(config, net, device, n_iter, n_iter_val, net_postfix="", train=True):
    """
    # load or new model, return net, optimizer, iter, 
    params:
        config: model params
        net: net work should be initialized
        device: 'cpu' or 'cuda'
        n_iter, n_iter_val: original iter
    return:
        net: model in gpu or cpu
        optimizer: the optimizer for the specific model
        n_iter -> int: resume training or new iter
        n_iter_val -> int: val iter
    """
    checkpoint_path = config["training"]["pretrained" + net_postfix]
    checkpoint_mode = "" if checkpoint_path[-3:] == "pth" else "full"
    if config["training"]["retrain" + net_postfix] == True:
        logging.info("New model")
    else:
        logging.info(
            "Loading net%s: path: %s, mode: %s"
            % (net_postfix, checkpoint_path, checkpoint_mode)
        )
        net, n_iter, n_iter_val = pretrainedLoader_net(
            net, n_iter, checkpoint_path, mode=checkpoint_mode, full_path=True
        )

    logging.info("+++[Train]+++ Let's use %d GPUs!" % torch.cuda.device_count())
    net = net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

        logging.info("+++[Train]+++ Paralell the network!")

    logging.info("+++[Train]+++ setting adam solver")
    optimizer = optim.Adam(net.parameters(), lr=config["training"]["learning_rate"])

    if (
        config["training"]["retrain" + net_postfix] == False
        and config["training"]["train_iter"] > 0
        and train
    ):
        logging.info(
            "Loading optimizer: path: %s, mode: %s" % (checkpoint_path, checkpoint_mode)
        )
        optimizer, n_iter, n_iter_val = pretrainedLoader_opt(
            optimizer, n_iter, checkpoint_path, mode=checkpoint_mode, full_path=True
        )

    if config["training"]["reset_iter" + net_postfix]:
        logging.info("reset iterations to 0")
        n_iter = 0
    logging.info("n_iter starts at %d" % n_iter)

    return net, optimizer, n_iter, n_iter_val


if __name__ == "__main__":
    # global var
    torch.set_default_tensor_type(torch.FloatTensor)
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        # level=logging.DEBUG,
    )

    # add parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # Training command
    p_train = subparsers.add_parser("train_good")
    p_train.add_argument("config", type=str)
    p_train.add_argument("exper_name", type=str)
    p_train.add_argument(
        "--eval",
        action="store_true",
        help="eval when training (can be used for validation)",
    )
    p_train.add_argument(
        "--test",
        action="store_true",
        help="use testing dataset instead of val dataset. Used for testing!",
    )
    p_train.add_argument(
        "--debug", action="store_true", default=False, help="turn on debuging mode"
    )
    p_train.set_defaults(func=train_good)

    # eval command --> dummy for tensorboard path
    p_train = subparsers.add_parser("eval_good")
    p_train.add_argument("config", type=str)
    p_train.add_argument("exper_name", type=str)
    p_train.add_argument(
        "--eval",
        action="store_true",
        help="eval when training (can be used for validation)",
    )
    p_train.add_argument(
        "--test",
        action="store_true",
        help="use testing dataset instead of val dataset. Used for testing!",
    )
    p_train.add_argument(
        "--debug", action="store_true", default=False, help="turn on debuging mode"
    )
    p_train.set_defaults(func=eval_good)

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # EXPER_PATH from settings.py
    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    os.makedirs(output_dir, exist_ok=True)

    # with capture_outputs(os.path.join(output_dir, 'log')):
    logging.info("Running command {}".format(args.command.upper()))
    args.func(config, output_dir, args)
