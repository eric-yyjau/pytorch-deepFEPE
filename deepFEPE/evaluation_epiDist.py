import argparse
import time
import csv
import yaml
import os
import logging
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm

from tensorboardX import SummaryWriter

## import from superpoint
from superpoint.utils.utils import (
    tensor2array,
    save_checkpoint,
    load_checkpoint,
    save_path_formatter,
    getWriterPath

)
from superpoint.utils.print_tool import datasize


from settings import EXPER_PATH
from utils.loader import dataLoader, modelLoader
# from utils.utils import getWriterPath

from superpoint.models.model_wrap import SuperPointFrontend_torch, PointTracker


def val_feature(config, output_dir, args):
    """
    1) input 2 images, output keypoints and correspondence

    :param config:
    :param output_dir:
    :param args:
    :return:
    """
    # config
    # device = torch.device("cpu")
    from superpoint.utils.var_dim import squeezeToNumpy

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info("train on device: %s", device)
    with open(os.path.join(output_dir, "config_sp.yml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    writer = SummaryWriter(getWriterPath(task=args.command, date=True))

    if config["training"]["reproduce"]:
        logging.info("reproduce = True")
        torch.manual_seed(0)
        np.random.seed(0)
        print(f"test random # : np({np.random.rand(1)}), torch({torch.rand(1)})")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ## save data
    from pathlib import Path

    # save_path = save_path_formatter(config, output_dir)

    from utils.loader import get_save_path

    save_path = get_save_path(output_dir)
    save_output = save_path / "../predictions"
    os.makedirs(save_output, exist_ok=True)

    # data loading
    from utils.loader import dataLoader as dataLoader

    # task = config['data']['dataset']
    val = 'test' if args.test else 'val' 
    val_shuffle = False if args.test else True
    train = False if args.test else True
    data = dataLoader(config, dataset=config["data"]["dataset"], train=train, warp_input=True, val=val, val_shuffle=val_shuffle)
    train_loader, test_loader = data["train_loader"], data["val_loader"]

    # data = dataLoader(config, dataset=config["data"]["dataset"], warp_input=True)
    # train_loader, test_loader = data["train_loader"], data["val_loader"]


    datasize(test_loader, config, tag="test")

    # model loading
    from utils.loader import get_module

    Val_model_heatmap = get_module("", config["front_end_model"])

    ## load pretrained
    val_agent = Val_model_heatmap(config["model"], device=device)
    val_agent.loadModel()

    ## tracker
    tracker = PointTracker(max_length=2, nn_thresh=val_agent.nn_thresh)

    ###### check!!!
    outputMatches = True
    count = 0
    subpixel = config["model"]["subpixel"]["enable"]
    patch_size = config["model"]["subpixel"]["patch_size"]
    # print("Do subpixel!!!")
    rand_noise = config["model"]["rand_noise"]["enable"]

    from utils.eval_tools import Result_processor

    result_dict_entry = [
        "epi_dist_mean_gt",
        "num_prob",
        "num_warped_prob",
        "num_matches",
        "mscores",
    ]
    result_processor = Result_processor(result_dict_entry)

    for i, sample in tqdm(enumerate(test_loader)):
        if config['training']['val_interval'] == -1:
            pass
        elif i>config['training']['val_interval']: 
            break

        # imgs_grey_float = [img_grey.float().cuda() / 255.0 for img_grey in imgs_grey]
        # img_0, img_1 = sample['imgs_grey'][0].unsqueeze(1), sample['imgs_grey'][1].unsqueeze(1)
        img_0, img_1 = (
            process_grey2tensor(sample["imgs_grey"][0], device=device),
            process_grey2tensor(sample["imgs_grey"][1], device=device),
        )

        # first image, no matches
        # img = img_0
        outs = get_pts_desc_from_agent(val_agent, img_0, subpixel, patch_size, device=device)
        pts, desc = outs["pts"], outs["desc"]  # pts: np [3, N]
        logging.info(f"pts: {pts.shape}, desc: {desc.shape}")
        
        if rand_noise:
            sigma = config["model"]["rand_noise"]["sigma"]
            noise = np.random.normal(loc=0.0, scale=sigma, size=pts.shape)
            pts[:2, :] += noise[:2, :]
            print("pts: ", pts[:, :3])

        if outputMatches == True:
            tracker.update(pts, desc)

        # save keypoints
        pred = {"image": squeezeToNumpy(img_0)}
        pred.update({"prob": pts.transpose(), "desc": desc.transpose()})
        pred.update({"num_prob": pts.shape[1]})
        logging.debug(f"1 -- pts: {pts.shape}, desc: {desc.shape}")

        # second image, output matches
        outs = get_pts_desc_from_agent(val_agent, img_1, subpixel, patch_size, device=device)
        pts, desc = outs["pts"], outs["desc"]
        # print(f"1 -- pts: {pts[:,:5]}, desc: {desc[:10, :5]}")
        # print(f"2 -- pts: {pts[:,:5]}, desc: {desc[:10, :5]}")

        if rand_noise:
            sigma = config["model"]["rand_noise"]["sigma"]
            noise = np.random.normal(loc=0.0, scale=sigma, size=pts.shape)
            pts[:2, :] += noise[:2, :]
            print("pts: ", pts[:, :3])

        if outputMatches == True:
            tracker.update(pts, desc)

        pred.update({"warped_image": squeezeToNumpy(img_1)})
        # print("total points: ", pts.shape)
        pred.update(
            {
                "warped_prob": pts.transpose(),
                "warped_desc": desc.transpose(),
                # "homography": squeezeToNumpy(sample["homography"]),
            }
        )
        logging.debug(f"2 -- pts: {pts.shape}")
        pred.update({"num_warped_prob": pts.shape[1]})

        # if subpixel:
        # pts = subpixel_fix(pts)

        if outputMatches == True:
            matches = tracker.get_matches()
            result = epi_dist_from_matches(
                matches.transpose()[np.newaxis, ...], sample, device, five_point=False
            )
            epi_dist_mean_gt = result["epi_dist_mean_gt"]
            logging.debug(f"epi_dist_mean_gt: {epi_dist_mean_gt.shape}")
            logging.info(f"matches: {matches.transpose().shape}, num_prob: {pred['num_prob']}, num_warped_prob: {pred['num_warped_prob']}")

            pred.update({"matches": matches.transpose()})
            pred.update({"num_matches": matches.shape[1]})
            pred.update({"epi_dist_mean_gt": epi_dist_mean_gt})
            mscores = tracker.get_mscores()
            # logging.info(f"tracker.get_mscores(): {mscores.shape}")
            pred.update({"mscores": mscores})


        """
        pred:
            'image': np(320,240)
            'prob' (keypoints): np (N1, 2)
            'desc': np (N2, 256)
            'warped_image': np(320,240)
            'warped_prob' (keypoints): np (N2, 2)
            'warped_desc': np (N2, 256)
            'homography': np (3,3)

        """
        # clean last descriptor
        tracker.clear_desc()

        # save data
        from pathlib import Path

        filename = str(count)
        path = Path(save_output, "{}.npz".format(filename))
        np.savez_compressed(path, **pred)
        # print("save: ", path)
        count += 1

        # process results
        result_processor.load_result(pred)
    params = {"inlier_ratio": config['evaluation']['inlier_thd']}
    # result_processor.output_result(['inlier_ratio'], **params)
    result_processor.inlier_ratio(
        "epi_dist_mean_gt", params["inlier_ratio"], if_print=True
    )
    result_processor.save_result(
        Path(save_output, "result_dict_all.npz"), "result_dict_all"
    )

    print(f"exp path: {save_output}, output pairs: {count}")


def process_grey2tensor(img, device="cpu"):
    img = img.float().to(device) / 255.0
    img = img.unsqueeze(-1).permute(0, 3, 1, 2)
    # print(f"img: {img[0,0,0,:10]}")
    return img


def get_pts_desc_from_agent(val_agent, img, subpixel, patch_size, device="cpu"):
    """
    pts: list [numpy (3, N)]
    desc: list [numpy (256, N)]
    """
    heatmap_batch = val_agent.run(img.to(device))  # heatmap: numpy [batch, 1, H, W]
    # heatmap to pts
    pts = val_agent.heatmap_to_pts()
    # print("pts from val_agent: ", pts[0].shape)
    if subpixel:
        pts = val_agent.soft_argmax_points(pts, patch_size=patch_size)
    # heatmap, pts to desc
    desc_sparse = val_agent.desc_to_sparseDesc()
    # print("pts[0]: ", pts[0].shape, ", desc_sparse[0]: ", desc_sparse[0].shape)
    # print("pts[0]: ", pts[0].shape)
    outs = {"pts": pts[0], "desc": desc_sparse[0]}
    return outs


## deprecated
def transpose_np_dict(outs):
    for entry in list(outs):
        outs[entry] = outs[entry].transpose()


def epi_dist_from_matches(matches_use, sample, device, five_point=False):
    """
    call val_rt function from sample and matches
    """
    from train_good_utils import val_rt

    Ks = sample["K"].to(device)
    E_gts, F_gts = sample["E"], sample["F"]
    x1, x2 = matches_use[:, :, :2], matches_use[:, :, 2:]  # [batch_size, N, 2(W, H)]
    scene_poses = sample[
        "relative_scene_poses"
    ]  # list of sequence_length tensors, which with size [batch_size, 4, 4]; the first being identity, the rest are [[R; t], [0, 1]]
    ## process variables
    K_np = Ks.cpu().numpy()
    # x1_np, x2_np = x1.detach().cpu().numpy(), x2.detach().cpu().numpy()
    x1_np, x2_np = x1, x2
    E_gt_np = E_gts.cpu().numpy()
    F_gt_np = F_gts.cpu().numpy()
    E_est_np = E_gt_np  ## copy gt
    F_est_np = F_gt_np
    logging.debug(f"x1_np: {x1_np.shape}, F_est_np: {F_est_np.shape}")

    delta_Rtijs_4_4 = scene_poses[
        1
    ].float()  # [batch_size, 4, 4], asserting we have 2 frames where scene_poses[0] are all identities
    delta_Rtijs_4_4_cpu_np = delta_Rtijs_4_4.cpu().numpy()

    idx = 0
    result = val_rt(
        idx,
        K_np[idx],
        x1_single_np=x1_np[idx],
        x2_single_np=x2_np[idx],
        E_est_np=E_est_np[[idx]],
        E_gt_np=E_gt_np[idx],
        F_est_np=F_est_np[idx],
        F_gt_np=F_gt_np[idx],
        delta_Rtijs_4_4_cpu_np=delta_Rtijs_4_4_cpu_np[idx],
        five_point=five_point,
        if_opencv=False,
    )

    error_Rt_estW, epi_dist_mean_estW, error_Rt_5point, epi_dist_mean_5point, error_Rt_gt, epi_dist_mean_gt = (
        result[0],
        result[1],
        result[2],
        result[3],
        result[4],
        result[5],
    )
    return {"epi_dist_mean_gt": epi_dist_mean_gt}


if __name__ == "__main__":
    # global var
    torch.set_default_tensor_type(torch.FloatTensor)
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG,
    )

    # add parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # export command
    p_train = subparsers.add_parser("val_feature")
    p_train.add_argument("config", type=str)
    p_train.add_argument("exper_name", type=str)
    p_train.add_argument("--correspondence", action="store_true")
    p_train.add_argument("--eval", action="store_true")
    p_train.add_argument("--test", action="store_true", help="use testing dataset instead of val dataset. Used for testing!")
    p_train.add_argument(
        "--debug", action="store_true", default=False, help="turn on debuging mode"
    )
    p_train.set_defaults(func=val_feature)

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f)
    print("check config!! ", config)
    # EXPER_PATH from settings.py
    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    os.makedirs(output_dir, exist_ok=True)

    # with capture_outputs(os.path.join(output_dir, 'log')):
    logging.info("Running command {}".format(args.command.upper()))
    args.func(config, output_dir, args)
