""" training script
Organized and documented by You-Yi on 07/07/2020.
This class is inheritaged from superpoint/Train_model_frontend.


"""

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm
import logging
from pathlib import Path

# from superpoint
from superpoint.utils.utils import save_checkpoint
from superpoint.Train_model_frontend import Train_model_frontend

## class specific
from deepFEPE.utils.loader import dataLoader, modelLoader  # pretrainedLoader
import deepFEPE.dsac_tools.utils_misc as utils_misc  # If cannot find: export KITTI_UTILS_PATH='/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils'
from deepFEPE.train_good_utils import (
    get_all_loss,
    val_rt,
    get_all_loss_DeepF,
    write_metrics_summary,
    # adjust_learning_rate,
    get_Rt_loss,
    mean_list,
    get_matches_from_SP,
)

from pebble import ProcessPool
import multiprocessing as mp
import copy

ratio_CPU = 0.4
default_number_of_process = int(ratio_CPU * mp.cpu_count())

##### other functions
def init_dict_of_lists(config, name_lists=[]):
    exps = {config["exps"]["our_name"]: [], config["exps"]["base_name"]: [], "gt": []}
    dict_of_lists = {
        "count": 0,
    }
    for i, en in enumerate(name_lists):
        dict_of_lists.update({f"{en}": copy.deepcopy(exps)})
    return dict_of_lists


def save_model(
    save_path, net, n_iter, n_iter_val, optimizer, loss, file_prefix=["superPointNet"]
):
    if getattr(net, "module", None) is not None:
        model_state_dict = net.module.state_dict()
    else:
        model_state_dict = net.state_dict()

    save_checkpoint(
        save_path,
        {
            "n_iter": n_iter + 1,
            "n_iter_val": n_iter_val + 1,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        n_iter,
        file_prefix=file_prefix,
    )
    logging.info("save model at training step: %d", n_iter)


##### end functions #####


class Train_model_pipeline(Train_model_frontend):
    """
    * training for deepFEPE
    """

    def __init__(self, config, save_path=".", args={}, device="cpu", verbose=False):
        self.config = config
        self.device = device
        self.save_path = save_path
        logging.info(f"set 'Train_model_pipeline' save_path: {save_path}")
        self.args = args
        # set training params
        self.clamp_cum = config["model"]["clamp_at"]
        self.if_SP = config["model"]["if_SP"]
        self.if_add_hist = False

    def get_train_params(self):
        config = self.config
        save_path = self.save_path
        train_params = {
            "save_path": save_path,
            "depth": config["model"]["depth"],
            "five_point": config["exps"]["five_point"],
            "if_quality": config["model"]["if_quality"],
            "if_img_des_to_pointnet": config["model"]["if_img_des_to_pointnet"],
            "if_learn_offsets": config["model"]["if_learn_offsets"],
            "if_tri_depth": config["model"]["if_tri_depth"],
            "if_sample_loss": config["model"]["if_sample_loss"],
        }
        return train_params

    @staticmethod
    def check_num_of_matches(sample, name="matches_good_unique_nums", thd=100):
        return sample[name].min() > thd

    @staticmethod
    def adjust_learning_rate(optimizer, epoch, lr_init, decay=0.9, step=100):
        """ used by Train_model_pipeline
        """
        lr = lr_init * (decay ** (epoch // step))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def get_learning_rate(self):
        """
        # update learning rate based on number of epochs.
        """
        config = self.config
        cur_lr = Train_model_pipeline.adjust_learning_rate(
            self.optimizer,
            self.epoch,
            config["training"]["learning_rate"],
            decay=config["training"]["lr_decay_rate"],
            step=config["training"]["lr_decay_step"],
        )
        self.cur_lr = cur_lr
        return cur_lr

    ##### set class parameters #####
    def set_params(self, n_iter, epoch, n_iter_val):
        self.n_iter = n_iter
        self.epoch = epoch
        self.n_iter_val = n_iter_val
        pass

    def set_nets(self, net, net_SP=None):
        self.net = net
        self.net_SP = net_SP
        pass

    def set_optimizers(self, optimizer, optimizer_SP=None):
        self.optimizer = optimizer
        self.optimizer_SP = optimizer_SP

    def set_SP_helpers(self, SP_processer, SP_tracker):
        self.SP_processer = SP_processer
        self.SP_tracker = SP_tracker

    ##### end set parameters ###

    def train_epoch(self, train=False):
        """ train or eval a epoch
        """
        # init params
        config = self.config
        writer = self.writer
        train_params = self.get_train_params()
        args = self.args
        # net, net_SP = self.net, self.net_SP
        optimizer, optimizer_SP = self.optimizer, self.optimizer_SP

        lr = self.get_learning_rate()
        logging.info(f"current learning rate: {lr}")

        running_losses = []
        self.save_lists = [
            "err_q",
            "err_t",
            "epi_dists",
            "relative_poses_cam",
            "relative_poses_body",
        ]
        dict_of_lists_in_train = init_dict_of_lists(config, self.save_lists)
        dict_of_lists_in_val = init_dict_of_lists(config, self.save_lists)
        if_val_in_train_trigger = False

        thd_corr = 300
        writer.add_scalar("training-lr", lr, self.n_iter)

        # Train one epoch
        for i, sample_train in tqdm(enumerate(self.train_loader)):
            # if training
            if train:
                # eval in training script
                if (
                    self.n_iter != 0
                    and self.n_iter % config["training"]["val_interval_in_train"] == 0
                ):
                    if_val_in_train_trigger = True
                if if_val_in_train_trigger:
                    logging.info(
                        "+++[Train]+++ Collecting training batch for %s at train step %d"
                        % (args.exper_name, self.n_iter)
                    )
                    self.net.eval()
                else:
                    self.net.train()

                # train one batch
                (
                    loss_train_out,
                    dict_of_lists_in_train,
                    clamp_cum,
                ) = self.train_val_batch(
                    train_params,
                    sample_train,
                    True,
                    if_val=if_val_in_train_trigger,
                    dict_of_lists=dict_of_lists_in_train,
                )

                if if_val_in_train_trigger:
                    if (
                        dict_of_lists_in_train["count"]
                        > config["training"]["val_batches"]
                    ):
                        dict_of_lists_in_train = self.flush_dict_of_lists(
                            writer, "training", self.n_iter, **dict_of_lists_in_train
                        )
                        if_val_in_train_trigger = False
                else:
                    # running_losses.append(loss_train_out)
                    print(self.n_iter, "%.8f" % loss_train_out)
                self.n_iter += 1

            # if testing
            if args.eval and self.n_iter % config["training"]["val_interval"] == 0:
                logging.info(
                    "+++[Val]+++ Validating %s at train step %d"
                    % (args.exper_name, self.n_iter)
                )
                self.net.eval()
                assert self.net.training == False
                for j, sample_val in tqdm(enumerate(self.val_loader)):
                    # if not self.check_num_of_matches(sample, thd=thd_corr): continue
                    logging.info("+++[Val]+++ Validating batch %d" % (j))
                    # logging.info(f"frame_id: {sample_val['frame_ids']}")
                    loss_val_out, dict_of_lists_in_val, _ = self.train_val_batch(
                        train_params, sample_val,
                        False, if_val=True, dict_of_lists=dict_of_lists_in_val,
                    )  ##### check: in order to align val and training
                    self.n_iter_val += 1
                    if config["training"]["val_batches"] != -1 and (
                        j > config["training"]["val_batches"]
                    ):  ##### check: how to limit the validation
                        break
                print(dict_of_lists_in_val.keys())

                ## save valdiation result (dict)
                if len(config["exps"]["filename"]) > 3:
                    # print(f"dict_of_lists_in_val: {dict_of_lists_in_val}")
                    def get_dict(key_layer1, key_layer2, dict_of_lists):
                        dict_of_array = {}
                        for k in key_layer1:
                            dict_of_array[k] = np.stack(dict_of_lists[k][key_layer2])
                        return dict_of_array

                    our_name, base_name = (
                        config["exps"]["our_name"],
                        config["exps"]["base_name"],
                    )

                    print(f'save dict_of_lists_in_val to {config["exps"]["filename"]}')
                    # save our results
                    dict_of_lists = get_dict(
                        self.save_lists, our_name, dict_of_lists_in_val
                    )
                    dict_of_lists["epi_dists"] = dict_of_lists["epi_dists"][:, :10]  ### only take part of it
                    np.savez(
                        f'{self.save_path[:-11]}/{our_name}_{config["exps"]["filename"]}',
                        **dict_of_lists,
                    )
                    # save base_name
                    dict_of_lists = get_dict(
                        self.save_lists, base_name, dict_of_lists_in_val
                    )
                    dict_of_lists["epi_dists"] = dict_of_lists["epi_dists"][:, :10]  ### only take part of it
                    np.savez(
                        f'{self.save_path[:-11]}/{base_name}_{config["exps"]["filename"]}',
                        **dict_of_lists,
                    )
                # output then flush
                dict_of_lists_in_val = self.flush_dict_of_lists(
                    writer, "validating", self.n_iter, **dict_of_lists_in_val
                )

            # epoch_loss = np.mean(np.asarray(running_losses))

            # training iterations
            self.epoch += 1
            if self.n_iter > config["training"]["train_iter"]:
                break
        return 0.0, self.clamp_cum, self.n_iter, self.n_iter_val

    def train_val_batch(
        self, train_params, sample, train, if_val=False, dict_of_lists=None,
    ):
        """ train or val one single batch
        
        """
        logging.debug(f"train_params: {train_params}")
        config = self.config
        device = self.device
        # net, net_SP = self.net, self.net_SP
        writer = self.writer

        if_train_SP = config["training"].get("train_SP", True)
        if_train_deepF = config["training"].get("train", True)
        task = "training" if train else "validating"
        n_iter_sample = self.n_iter if train else self.n_iter_val

        Ks = sample["K"].to(device)  # [batch_size, 3, 3]
        K_invs = sample["K_inv"].to(device)  # [batch_size, 3, 3]
        batch_size = Ks.size(0)
        scene_names = sample["scene_name"]
        frame_ids = sample["frame_ids"]
        scene_poses = sample[
            "relative_scene_poses"
        ]  # list of sequence_length tensors, which with size [batch_size, 4, 4]; the first being identity, the rest are [[R; t], [0, 1]]
        if config["data"]["read_what"]["with_X"] and batch_size == 1:
            Xs = sample[
                "Xs"
            ]  # list of [batch_size, 3, Ni]; only support batch_size=1 because of variable points Ni for each sample
        # sift_kps, sift_deses = sample['sift_kps'], sample['sift_deses']
        assert sample["get_flags"]["have_matches"][
            0
        ].numpy(), "Did not find the corres files!"
        matches_all, matches_good = sample["matches_all"], sample["matches_good"]

        if train_params["if_quality"]:
            quality_good = sample["quality_good"]
        # if train_params['if_des']:
        #     des_good = sample['des_good']

        delta_Rtijs_4_4 = scene_poses[
            1
        ].float()  # [batch_size, 4, 4], asserting we have 2 frames where scene_poses[0] are all identities
        # E_gts, F_gts = utils_F._E_F_from_Rt(delta_Rtijs_4_4[:, :3, :3], delta_Rtijs_4_4[:, :3, 3:4], Ks, tensor_input=True)
        E_gts, F_gts = sample["E"], sample["F"]
        # pts1_virt_normalizedK, pts2_virt_normalizedK = sample['pts1_virt_normalized'].cuda(), sample['pts2_virt_normalized'].cuda()

        if config["model"]["if_lidar_corres"]:
            pts1_virt_ori, pts2_virt_ori = (
                sample["pts1_velo"].cuda(),
                sample["pts2_velo"].cuda(),
            )
            logging.warning("Using lidar virtual corres for loss!")
        else:
            pts1_virt_ori, pts2_virt_ori = (
                sample["pts1_virt"].cuda(),
                sample["pts2_virt"].cuda(),
            )

        ## get matches (from processed data or superpoint)
        if self.if_SP:
            if not if_train_SP:
                with torch.no_grad():
                    self.net_SP.eval()
                    data = get_matches_from_SP(
                        sample["imgs_grey"], self.net_SP, self.SP_processer, self.SP_tracker
                    )
            else:
                data = get_matches_from_SP(
                    sample["imgs_grey"], self.net_SP, self.SP_processer, self.SP_tracker
                )

            # {'xs': xs, 'offsets': offsets, 'quality': quality, 'num_matches': num_matches}
            xs, offsets, quality = data["xs"], data["offsets"], data["quality"]
            num_matches = data["num_matches"]

            # matches_use = (xs + offsets).detach(); logging.warning("gradient of superpoint is detached")
            matches_use = xs + offsets

            quality_use = quality
        else:
            if_SIFT = True
            if if_SIFT:
                ## run sift on the fly
                pass
            # Get and Normalize points
            matches_use = matches_good  # [SWITCH!!!]
            if train_params["if_quality"]:
                quality_use = quality_good.cuda()  # [SWITCH!!!]
            else:
                quality_use = None
        N_corres = matches_use.shape[1]  # 1311 for matches_good, 2000 for matches_all
        x1, x2 = (
            matches_use[:, :, :2],
            matches_use[:, :, 2:],
        )  # [batch_size, N, 2(W, H)]

        x1_normalizedK = utils_misc._de_homo(
            torch.matmul(
                torch.inverse(Ks), utils_misc._homo(x1.to(device)).transpose(1, 2)
            ).transpose(1, 2)
        )  # [batch_size, N, 2(W, H)], min/max_X=[-W/2/f, W/2/f]
        x2_normalizedK = utils_misc._de_homo(
            torch.matmul(
                torch.inverse(Ks), utils_misc._homo(x2.to(device)).transpose(1, 2)
            ).transpose(1, 2)
        )  # [batch_size, N, 2(W, H)], min/max_X=[-W/2/f, W/2/f]
        matches_use_normalizedK = torch.cat((x1_normalizedK, x2_normalizedK), 2)

        matches_use_ori = torch.cat((x1, x2), 2).cuda()

        # Get image feats
        if config["model"]["if_img_feat"]:
            imgs = sample["imgs"]  # [batch_size, H, W, 3]
            imgs_stack = ((torch.cat(imgs, 3).float() - 127.5) / 127.5).permute(
                0, 3, 1, 2
            )

        qs_scene = sample["q_scene"].cuda()  # [B, 4, 1]
        ts_scene = sample["t_scene"].cuda()  # [B, 3, 1]
        qs_cam = sample["q_cam"].cuda()  # [B, 4, 1]
        ts_cam = sample["t_cam"].cuda()  # [B, 3, 1]

        t_scene_scale = torch.norm(ts_scene, p=2, dim=1, keepdim=True)

        # Make data batch
        data_batch = {
            "matches_xy": matches_use_normalizedK,
            "matches_xy_ori": matches_use_ori,
            "quality": quality_use,
            "x1_normalizedK": x1_normalizedK,
            "x2_normalizedK": x2_normalizedK,
            "Ks": Ks,
            "K_invs": K_invs,
            "des1": None,
            "des2": None,
            "matches_good_unique_nums": sample["matches_good_unique_nums"],
            "t_scene_scale": t_scene_scale,
            "frame_ids": sample["frame_ids"],
        }
        # if train_params['if_des']:
        #     data_batch['des1'] = des_good[:, :, :128].cuda()
        #     data_batch['des2'] = des_good[:, :, 128:].cuda()
        if config["model"]["if_img_feat"]:
            data_batch["imgs_stack"] = imgs_stack.cuda()

        # loss_params = {'model': config['model']['name'], 'clamp_at':config['model']['clamp_at'], 'depth': config['model']['depth']}
        loss_params = {
            "model": config["model"]["name"],
            "clamp_at": self.clamp_cum,
            "depth": config["model"]["depth"],
            "good_num": config["data"]["good_num"],
            "if_img_feat": config["model"]["if_img_feat"],
            "matches_good_unique_nums": sample["matches_good_unique_nums"],
            "topK": 8,
            "if_sample_loss": config["model"]["if_sample_loss"],
            "if_tri_depth": config["model"]["if_tri_depth"],
        }

        get_residual_summaries = False
        balance_q = config["model"].get("balance_q", 10.0)
        balance_t = config["model"].get("balance_t", 1.0)
        balance_F = config["model"].get("balance_F", 100.0)
        balance_select_F = config["model"].get("balance_select_F", 0.1)
        clamp_q_params = config["training"].get("clamp_q_params", [0.1, 0.01, 0.001])
        clamp_t_params = config["training"].get("clamp_t_params", [0.5, 0.1, 0.03])

        if self.n_iter < config["training"].get("clamp_iter1", 1600):
            # < 1600
            loss_q_clamp = clamp_q_params[0]
            loss_t_clamp = clamp_t_params[0]
            # loss_t_clamp = 0.5
        elif self.n_iter < config["training"].get("clamp_iter2", 3800):
            # < 3800
            loss_q_clamp = clamp_q_params[1]
            loss_t_clamp = clamp_t_params[1]
            # loss_t_clamp = 0.5
        else:
            #
            loss_q_clamp = clamp_q_params[2]
            loss_t_clamp = clamp_t_params[2]
            # loss_q_clamp = 0.01
            # loss_t_clamp = 0.5  # 0.2

        loss_F_before_iter = -1
        reg = 0.0

        if train:
            outs = self.net(data_batch)
            # loss, loss_layers, loss_E, E_ests, logits_weights, logits = get_all_loss(
            #     outs, x1_normalizedK, x2_normalizedK, pts1_virt_normalizedK, pts2_virt_normalizedK, Ks, E_gts, loss_params)

            ## get losses (deepF, f-loss)
            (
                losses_dict,
                E_ests,
                F_ests,
                logits_weights,
                residual_norm_layers,
                residual_norm_max_layers,
                E_ests_layers,
            ) = get_all_loss_DeepF(
                outs,
                pts1_virt_ori,
                pts2_virt_ori,
                Ks,
                loss_params,
                get_residual_summaries=get_residual_summaries,
            )

            loss_F = losses_dict["loss_F"]
            loss_epi_res = losses_dict["loss_epi_res"]
            loss_layers = losses_dict["loss_layers"]
            # logging.info(f"loss_F: {loss_F}, loss_epi_res: {loss_epi_res}, loss_layers: {loss_layers}")
            if get_residual_summaries:
                loss_residual = losses_dict["loss_residual"]
                loss_residual_topK = losses_dict["loss_residual_topK"]
                loss_regW_clip = losses_dict["loss_regW_clip"]
                loss_regW_entro = losses_dict["loss_regW_entro"]
                loss_regW_entro_topK = losses_dict["loss_regW_entro_topK"]

            loss = loss_F
            loss_min = losses_dict["loss_min_batch"].min()

            if loss_params["if_tri_depth"]:
                loss_depth_mean = losses_dict["loss_depth_mean"]
                loss = loss + loss_depth_mean

            if train_params["if_sample_loss"]:
                print("Added loss_selected_F")
                loss_selected_F = losses_dict["loss_selected_F"]
                loss_selected_layers = losses_dict["loss_selected_layers"]
                loss += loss_selected_F * balance_select_F

            ## get pose loss (pose-loss)
            if config["model"]["if_qt_loss"]:
                # to_cpu = lambda x: x.cpu()
                ## detach x1, x2 to calculate loss
                geo_errors_dict = get_Rt_loss(
                    E_ests_layers,
                    sample["K"],
                    x1.detach().cpu(),
                    x2.detach().cpu(),
                    delta_Rtijs_4_4,
                    qs_cam,
                    ts_cam,
                    device=self.device
                )
                R_angle_error_mean = geo_errors_dict["R_angle_error_mean"]
                t_angle_error_mean = geo_errors_dict["t_angle_error_mean"]
                R_angle_error_list = geo_errors_dict["R_angle_error_list"]
                t_angle_error_list = geo_errors_dict["t_angle_error_list"]
                q_l2_error_mean = geo_errors_dict["q_l2_error_mean"]
                t_l2_error_mean = geo_errors_dict["t_l2_error_mean"]
                q_l2_error_list = geo_errors_dict["q_l2_error_list"]
                t_l2_error_list = geo_errors_dict["t_l2_error_list"]
                R_angle_error_layers_list = geo_errors_dict[
                    "R_angle_error_layers_list"
                ]  # [[B], [B], ...]
                t_angle_error_layers_list = geo_errors_dict["t_angle_error_layers_list"]
                q_l2_error_layers_list = geo_errors_dict["q_l2_error_layers_list"]
                t_l2_error_layers_list = geo_errors_dict["t_l2_error_layers_list"]

                # loss_q = sum(torch.clamp(q_l2_error_list, 0., 5))/len(q_l2_error_list)
                loss_q = q_l2_error_mean
                # loss_q = sum(torch.clamp(R_angle_error_list, 0., 4.))/len(q_l2_error_list) * 10
                loss_t = sum(t_l2_error_list) / len(t_l2_error_list)
                # loss_t = sum(torch.clamp(t_l2_error_list, 0., 0.02))/len(t_l2_error_list)
                if self.n_iter < loss_F_before_iter:
                    print("--DeepF loss--")
                    loss = loss_F
                else:
                    print("--qt loss--")
                    loss_q = torch.clamp(
                        torch.stack(q_l2_error_layers_list), 0.0, loss_q_clamp
                    ).mean()
                    loss_t = torch.clamp(
                        torch.stack(t_l2_error_layers_list), 0.0, loss_t_clamp
                    ).mean()
                    loss = loss_q * balance_q + loss_t * balance_t
                    # loss += loss_F * balance_F
                    print(loss_q.item(), loss_t.item())

            # zero the parameter gradients, train the models!!
            if if_train_deepF:
                self.optimizer.zero_grad()
            if self.if_SP and if_train_SP:
                self.optimizer_SP.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(net.parameters(), config['training']['gradient_clip'])
            # nn.utils.clip_grad_norm_(net.parameters(), 0.1)
            def check_skip_condition(skip_condition, loss):
                if skip_condition is None:
                    return False
                if skip_condition.get("enable", False):
                    if loss <= skip_condition["params"]["epi_min"]:
                        return True
                return False

            def add_msg_to_log(log_file, msg):
                with open(log_file, "a") as log_f:
                    log_f.write(f"{msg}\n")
                pass

            if_skip = config["training"].get("skip_optimizer", None)

            ## add msgs to log file
            # loss_min_batch = losses_dict["loss_min_batch"]
            for i, (frame_id, loss_temp) in enumerate(
                zip(np.array(data_batch["frame_ids"]).T, losses_dict["loss_min_batch"])
            ):
                if check_skip_condition(if_skip, loss_temp):
                    add_msg_to_log(
                        Path(self.save_path) / "log.txt",
                        f"loss is lower than epi_min. loss: {loss_min}, frame_id: {frame_id}",
                    )
            ## for debugging
            skip = False
            if if_skip is not None:
                logging.info(
                    f"loss_min: {loss_min}, loss_batch: {losses_dict['loss_min_batch']}"
                )
                skip = check_skip_condition(if_skip, loss_min)
            
            ## optimizer step
            if skip or not if_train_deepF:
                logging.info(f"skip optimizing: {loss}")
                logging.info(
                    f"loss_min: {loss_min}, loss_batch: {losses_dict['loss_min_batch']}"
                )
                logging.info(f"frames: {data_batch['frame_ids']}")
                # logging.info("optimizer!!")
            else:
                self.optimizer.step()
            if self.if_SP and if_train_SP:
                logging.info("training SP")
                self.optimizer_SP.step()

        # testing
        else:
            with torch.no_grad():
                outs = self.net(data_batch)
                # loss, loss_layers, loss_E, E_ests, logits_weights, logits = get_all_loss(
                #     outs, x1_normalizedK, x2_normalizedK, pts1_virt_normalizedK, pts2_virt_normalizedK, Ks, E_gts, loss_params)

                ## get losses (f-loss)
                (
                    losses_dict,
                    E_ests,
                    F_ests,
                    logits_weights,
                    residual_norm_layers,
                    residual_norm_max_layers,
                    E_ests_layers,
                ) = get_all_loss_DeepF(
                    outs,
                    pts1_virt_ori,
                    pts2_virt_ori,
                    Ks,
                    loss_params,
                    get_residual_summaries=get_residual_summaries,
                )
                loss_F = losses_dict["loss_F"]
                loss_epi_res = losses_dict["loss_epi_res"]
                loss_layers = losses_dict["loss_layers"]
                if get_residual_summaries:
                    loss_residual = losses_dict["loss_residual"]
                    loss_residual_topK = losses_dict["loss_residual_topK"]
                    loss_regW_clip = losses_dict["loss_regW_clip"]
                    loss_regW_entro = losses_dict["loss_regW_entro"]
                    loss_regW_entro_topK = losses_dict["loss_regW_entro_topK"]

                loss = loss_F
                loss += reg

                if loss_params["if_tri_depth"]:
                    loss_depth_mean = losses_dict["loss_depth_mean"]
                    loss = loss + loss_depth_mean

                if train_params["if_sample_loss"]:
                    loss_selected_F = losses_dict["loss_selected_F"]
                    loss_selected_layers = losses_dict["loss_selected_layers"]
                    loss += loss_selected_F * balance_select_F

                ## pose loss (pose-loss)
                if config["model"]["if_qt_loss"]:
                    geo_errors_dict = get_Rt_loss(
                        E_ests_layers,
                        sample["K"],
                        x1.cpu(),
                        x2.cpu(),
                        delta_Rtijs_4_4,
                        qs_cam,
                        ts_cam,
                        device=self.device
                    )
                    R_angle_error_mean = geo_errors_dict["R_angle_error_mean"]
                    t_angle_error_mean = geo_errors_dict["t_angle_error_mean"]
                    R_angle_error_list = geo_errors_dict["R_angle_error_list"]
                    t_angle_error_list = geo_errors_dict["t_angle_error_list"]
                    q_l2_error_mean = geo_errors_dict["q_l2_error_mean"]
                    t_l2_error_mean = geo_errors_dict["t_l2_error_mean"]
                    q_l2_error_list = geo_errors_dict["q_l2_error_list"]
                    t_l2_error_list = geo_errors_dict["t_l2_error_list"]
                    R_angle_error_layers_list = geo_errors_dict[
                        "R_angle_error_layers_list"
                    ]  # [[B], [B], ...]
                    t_angle_error_layers_list = geo_errors_dict[
                        "t_angle_error_layers_list"
                    ]
                    q_l2_error_layers_list = geo_errors_dict["q_l2_error_layers_list"]
                    t_l2_error_layers_list = geo_errors_dict["t_l2_error_layers_list"]

                    # loss_q = sum(torch.clamp(q_l2_error_list, 0., 5))/len(q_l2_error_list)
                    loss_q = q_l2_error_mean
                    # loss_q = sum(torch.clamp(R_angle_error_list, 0., 4.))/len(q_l2_error_list) * 10
                    loss_t = sum(t_l2_error_list) / len(t_l2_error_list)
                    # loss_t = sum(torch.clamp(t_l2_error_list, 0., 0.02))/len(t_l2_error_list)
                    if self.n_iter < loss_F_before_iter:
                        print("--DeepF loss--")
                        loss = loss_F
                    else:
                        print("--qt loss--")
                        loss_q = torch.clamp(
                            torch.stack(q_l2_error_layers_list), 0.0, loss_q_clamp
                        ).mean()
                        loss_t = torch.clamp(
                            torch.stack(t_l2_error_layers_list), 0.0, loss_t_clamp
                        ).mean()
                        loss = loss_q * balance_q + loss_t * balance_t
                        # loss += loss_F * balance_F
                        print(loss_q.item(), loss_t.item())

        # print(loss_F)
        ## put losses to the writer
        if loss_params["if_tri_depth"]:
            writer.add_scalar(
                task + "-loss/loss_depth_mean", loss_depth_mean, n_iter_sample
            )
        if config["model"]["if_qt_loss"]:
            writer.add_scalar(task + "-clamping/loss_q", loss_q_clamp, n_iter_sample)
            writer.add_scalar(task + "-clamping/loss_t", loss_t_clamp, n_iter_sample)

        writer.add_scalar(task + "-loss/loss", loss, n_iter_sample)
        writer.add_scalar(task + "-loss/loss_all", loss_F, n_iter_sample)
        writer.add_scalar(task + "-loss/loss_epi_res", loss_epi_res, n_iter_sample)
        for idx, loss_layer in enumerate(loss_layers):
            writer.add_scalar(
                task + "-loss/loss_layer_%d" % (idx - train_params["depth"]),
                loss_layer,
                n_iter_sample,
            )

        if train_params["if_sample_loss"]:
            writer.add_scalar(
                task + "-loss/loss_selected_F", loss_selected_F, n_iter_sample
            )
            for idx, loss_selected_layer in enumerate(loss_selected_layers):
                writer.add_scalar(
                    task
                    + "-loss/loss_selected_layer_%d" % (idx - train_params["depth"]),
                    loss_selected_layer,
                    n_iter_sample,
                )

        if get_residual_summaries:
            writer.add_scalar(
                task + "-loss/loss_residual", loss_residual, n_iter_sample
            )
            writer.add_scalar(
                task + "-loss/loss_residual_topK", loss_residual_topK, n_iter_sample
            )
            writer.add_scalar(task + "-reg/loss_regALL", reg, n_iter_sample)

            ## Regs
            writer.add_scalar(
                task + "-reg/loss_regW_clip", loss_regW_clip, n_iter_sample
            )
            writer.add_scalar(
                task + "-reg/loss_regW_entropy", loss_regW_entro, n_iter_sample
            )
            writer.add_scalar(
                task + "-reg/loss_regW_entropy_topk",
                loss_regW_entro_topK,
                n_iter_sample,
            )
            writer.add_scalar(
                task + "-debug/logits_weights_MAX",
                np.amax(logits_weights.detach().cpu().numpy()),
                n_iter_sample,
            )

            for idx, (residual_norm_layer, residual_norm_max_layer) in enumerate(
                zip(residual_norm_layers, residual_norm_max_layers)
            ):
                writer.add_scalar(
                    task
                    + "-loss_residual_norm/residual_norm_layer_%d"
                    % (idx - train_params["depth"]),
                    residual_norm_layer,
                    n_iter_sample,
                )
                writer.add_scalar(
                    task
                    + "-debug/residual_norm_max_layer_%d"
                    % (idx - train_params["depth"]),
                    residual_norm_max_layer,
                    n_iter_sample,
                )

        ## Geo losses
        if self.n_iter % 10 == 0:
            if config["model"]["if_qt_loss"]:
                np.set_printoptions(precision=4, suppress=True)
                print(
                    "````q degre",
                    np.amax(R_angle_error_layers_list[-1]),
                    R_angle_error_layers_list[-1],
                )
                print(
                    "----q error",
                    np.amax(q_l2_error_layers_list[-1].detach().cpu().numpy()),
                    q_l2_error_layers_list[-1].detach().cpu().numpy(),
                )
                print(
                    "++++t degre",
                    np.amax(t_angle_error_layers_list[-1]),
                    t_angle_error_layers_list[-1],
                )
                print(
                    "----t error",
                    np.amax(t_l2_error_layers_list[-1].detach().cpu().numpy()),
                    t_l2_error_layers_list[-1].detach().cpu().numpy(),
                )
                writer.add_scalar(
                    task + "-loss_geo/R_angle_mean", R_angle_error_mean, n_iter_sample
                )
                writer.add_scalar(
                    task + "-loss_geo/t_angle_mean", t_angle_error_mean, n_iter_sample
                )
                # writer.add_scalar(task + '-loss_geo/t_l2_mean', t_l2_error_mean, n_iter_sample)
                # writer.add_scalar(task + '-loss_geo/q_l2_mean', q_l2_error_mean, n_iter_sample)
                R_angle_errors = R_angle_error_layers_list[-1].flatten()
                if self.if_add_hist:
                    writer.add_histogram(
                        task + "-loss_geo/R_angle_error", R_angle_errors, n_iter_sample
                    )
                    writer.add_histogram(
                        task + "-loss_geo/R_angle_error-Clip10.degree",
                        np.clip(R_angle_errors, 0.0, 10.0),
                        n_iter_sample,
                    )
                    writer.add_histogram(
                        task + "-loss_geo/R_angle_error-Thres0.05.degree",
                        np.hstack((R_angle_errors[R_angle_errors < 0.05], np.zeros(1))),
                        n_iter_sample,
                    )
                    t_angle_errors = t_angle_error_layers_list[-1].flatten()
                    writer.add_histogram(
                        task + "-loss_geo/t_angle_error", t_angle_errors, n_iter_sample
                    )
                    writer.add_histogram(
                        task + "-loss_geo/t_angle_error-Clip10.degree",
                        np.clip(t_angle_errors, 0.0, 10.0),
                        n_iter_sample,
                    )
                    writer.add_histogram(
                        task + "-loss_geo/t_angle_error-Thres0.5.degree",
                        np.hstack((t_angle_errors[t_angle_errors < 0.5], np.zeros(1))),
                        n_iter_sample,
                    )
                writer.add_scalar(
                    task + "-loss_geo/R_angle_error-MAX.degree",
                    np.max(R_angle_errors),
                    n_iter_sample,
                )
                writer.add_scalar(
                    task + "-loss_geo/t_angle_error-MAX.degree",
                    np.amax(t_angle_errors),
                    n_iter_sample,
                )

                for idx, (q_l2_error_layer_list, t_l2_error_layer_list) in enumerate(
                    zip(q_l2_error_layers_list, t_l2_error_layers_list)
                ):
                    if self.if_add_hist:
                        writer.add_histogram(
                            task + "-loss_geo/q_l2_error_%d" % idx,
                            q_l2_error_layer_list.clone().cpu().data.numpy(),
                            n_iter_sample,
                        )
                        writer.add_histogram(
                            task + "-loss_geo/t_l2_error_%d" % idx,
                            t_l2_error_layer_list.clone().cpu().data.numpy(),
                            n_iter_sample,
                        )
                        writer.add_histogram(
                            task + "-loss_geo/q_l2_error_%d-Thres" % idx,
                            np.clip(
                                q_l2_error_layer_list.clone().cpu().data.numpy(),
                                0.0,
                                loss_q_clamp,
                            ),
                            n_iter_sample,
                        )
                        writer.add_histogram(
                            task + "-loss_geo/t_l2_error_%d-Thres" % idx,
                            np.clip(
                                t_l2_error_layer_list.clone().cpu().data.numpy(),
                                0.0,
                                loss_t_clamp,
                            ),
                            n_iter_sample,
                        )
                    writer.add_scalar(
                        task + "-loss_geo/loss_q_l2_%d" % idx,
                        torch.clamp(q_l2_error_layer_list, 0.0, loss_q_clamp).mean(),
                        n_iter_sample,
                    )
                    writer.add_scalar(
                        task + "-loss_geo/loss_t_l2_%d" % idx,
                        torch.clamp(t_l2_error_layer_list, 0.0, loss_t_clamp).mean(),
                        n_iter_sample,
                    )
                writer.add_scalar(task + "-loss_geo/loss_q_l2", loss_q, n_iter_sample)
                writer.add_scalar(task + "-loss_geo/loss_t_l2", loss_t, n_iter_sample)

            ## Summaries
            if self.if_add_hist:
                writer.add_histogram(
                    task + "-logits_weights",
                    logits_weights.clone().cpu().data.numpy(),
                    n_iter_sample,
                )
                if train_params["if_learn_offsets"]:
                    writer.add_histogram(
                        task + "-offsets",
                        outs["offsets"].clone().cpu().data.numpy(),
                        n_iter_sample,
                    )
                if train_params["if_tri_depth"]:
                    writer.add_histogram(
                        task + "-tri_depths",
                        outs["tri_depths"].clone().cpu().data.numpy(),
                        n_iter_sample,
                    )
        
        if if_val:
            E_recover_110_lists = []
            for E_recover in E_ests.detach().cpu():
                FU, FD, FV = torch.svd(E_recover, some=True)
                # print('[info.Debug @_E_from_XY] Singular values for recovered E(F):\n', FD.numpy())
                S_110 = torch.diag(
                    torch.tensor([1.0, 1.0, 0.0], dtype=FU.dtype, device=FU.device)
                )
                E_recover_110 = torch.mm(FU, torch.mm(S_110, FV.t()))
                E_recover_110_lists.append(E_recover_110)
            E_ests_110 = torch.stack(E_recover_110_lists)

            K_np = Ks.cpu().numpy()
            x1_np, x2_np = x1.detach().cpu().numpy(), x2.detach().cpu().numpy()
            E_est_np = E_ests_110.detach().cpu().numpy()
            E_gt_np = E_gts.cpu().numpy()
            F_est_np = F_ests.detach().cpu().numpy()
            F_gt_np = F_gts.cpu().numpy()
            delta_Rtijs_4_4_cpu_np = delta_Rtijs_4_4.cpu().numpy()

            if config["model"]["if_img_feat"]:
                imgWeights_mean = [
                    torch.clamp(
                        1.0 / torch.sigmoid(outs["imgWeights"][idx][:, 0, :, :]),
                        0.0,
                        200.0,
                    )
                    for idx in range(2)
                ]
                im_weight_mean_i_all = (
                    imgWeights_mean[0].cpu().detach().squeeze().numpy()
                )  # [B, H, W]
                im_weight_mean_j_all = (
                    imgWeights_mean[1].cpu().detach().squeeze().numpy()
                )  # [B, H, W]
                # im_feat_i_all = outs['imgFeats'][0].cpu().detach().squeeze().numpy() #[B, D, H, W]
                # im_feat_j_all = outs['imgFeats'][1].cpu().detach().squeeze().numpy() #[B, D, H, W]
                im_weight_mean_i_all = im_weight_mean_i_all / np.max(
                    im_weight_mean_i_all
                )
                im_weight_mean_j_all = im_weight_mean_j_all / np.max(
                    im_weight_mean_j_all
                )

                for idx in range(batch_size // 2):
                    writer.add_image(
                        task + "-imgs_sample_i/%d" % idx,
                        np.transpose(imgs[0][idx], (2, 0, 1)),
                        n_iter_sample,
                    )
                    writer.add_image(
                        task + "-imgs_sample_j/%d" % idx,
                        np.transpose(imgs[1][idx], (2, 0, 1)),
                        n_iter_sample,
                    )
                    im_weight_mean_i_rgb = cv2.cvtColor(
                        im_weight_mean_i_all[idx], cv2.COLOR_GRAY2RGB
                    )
                    im_weight_mean_j_rgb = cv2.cvtColor(
                        im_weight_mean_j_all[idx], cv2.COLOR_GRAY2RGB
                    )
                    writer.add_image(
                        task + "-weights_mean_sample_i/%d" % idx,
                        np.transpose(im_weight_mean_i_rgb, (2, 0, 1)),
                        n_iter_sample,
                    )
                    writer.add_image(
                        task + "-weights_mean_sample_j/%d" % idx,
                        np.transpose(im_weight_mean_j_rgb, (2, 0, 1)),
                        n_iter_sample,
                    )
                    im_weight_mean_diff_rgb = np.abs(
                        im_weight_mean_i_rgb - im_weight_mean_j_rgb
                    )
                    writer.add_image(
                        task + "-weights_mean_sample_ijDiff/%d" % idx,
                        np.transpose(
                            im_weight_mean_diff_rgb / np.amax(im_weight_mean_diff_rgb),
                            (2, 0, 1),
                        ),
                        n_iter_sample,
                    )

                # if config['model']['if_img_des_to_pointnet']:
                #     writer.add_histogram('feats_im1', outs['featsN_im1'][0].clone().cpu().data.numpy(), n_iter_sample)
                #     writer.add_histogram('feats_im1_channel0', outs['featsN_im1'][0][0].clone().cpu().data.numpy(), n_iter_sample)
                writer.add_histogram(
                    "quality_batch",
                    outs["quality"].clone().cpu().data.numpy(),
                    n_iter_sample,
                )
                # writer.add_histogram('reses-layer0_batch', outs['reses'][0].clone().cpu().data.numpy(), n_iter_sample)

            ## multi-thread for validation
            with ProcessPool(max_workers=default_number_of_process) as pool:
                tasks = pool.map(
                    val_rt, # function called
                    range(batch_size),
                    [K_np[idx] for idx in range(batch_size)],
                    [x1_np[idx] for idx in range(batch_size)],
                    [x2_np[idx] for idx in range(batch_size)],
                    [E_est_np[idx] for idx in range(batch_size)],
                    [E_gt_np[idx] for idx in range(batch_size)],
                    [F_est_np[idx] for idx in range(batch_size)],
                    [F_gt_np[idx] for idx in range(batch_size)],
                    [delta_Rtijs_4_4_cpu_np[idx] for idx in range(batch_size)],
                    [train_params["five_point"]] * batch_size,
                )
                try:
                    # get estimated R,t and error
                    for i, result in enumerate(tasks.result()):
                        (
                            error_Rt_estW,
                            epi_dist_mean_estW,
                            error_Rt_5point,
                            epi_dist_mean_5point,
                            error_Rt_gt,
                            epi_dist_mean_gt,
                        ) = (
                            result[0],
                            result[1],
                            result[2],
                            result[3],
                            result[4],
                            result[5],
                        )
                        M_estW = result[7]
                        M_opencv = result[8]
                        M_estW = utils_misc.Rt_pad(M_estW)
                        M_opencv = utils_misc.Rt_pad(M_opencv)
                        if error_Rt_estW and error_Rt_5point:
                            our_name, base_name = (
                                config["exps"]["our_name"],
                                config["exps"]["base_name"],
                            )
                            dict_of_lists["err_q"][our_name].append(error_Rt_estW[0])
                            dict_of_lists["err_t"][our_name].append(error_Rt_estW[1])
                            dict_of_lists["epi_dists"][our_name].append(
                                np.expand_dims(epi_dist_mean_estW, -1)
                            )

                            Rt_cam2_gt_np = sample["Rt_cam2_gt"].numpy()
                            logging.info(f"Rt_cam2_gt_np: {Rt_cam2_gt_np.shape}, M_estW: {M_estW.shape}")
                            # logging.info(f"Rt_cam2_gt_np: {Rt_cam2_gt_np[0]}")
                            def relative_pose_cam_to_body(
                                relative_scene_pose, Rt_cam2_gt
                            ):
                                """ transform the camera pose from camera coordinate to body coordinate
                                """
                                relative_scene_pose = (
                                    np.linalg.inv(Rt_cam2_gt)
                                    @ relative_scene_pose
                                    @ Rt_cam2_gt
                                )
                                return relative_scene_pose

                            M_estW_body = relative_pose_cam_to_body(
                                M_estW, Rt_cam2_gt_np[i]
                            )
                            dict_of_lists["relative_poses_cam"][our_name].append(M_estW)
                            # print(f"M_estW: {M_estW}")
                            # print(f"M_estW_body: {M_estW_body}")
                            # save estimated poses
                            dict_of_lists["relative_poses_body"][our_name].append(
                                M_estW_body
                            )

                            ### baseline and gt
                            dict_of_lists["err_q"][base_name].append(error_Rt_5point[0])
                            dict_of_lists["err_t"][base_name].append(error_Rt_5point[1])
                            dict_of_lists["epi_dists"][base_name].append(
                                np.expand_dims(epi_dist_mean_5point, -1)
                            )
                            ### save gt poses in base_name
                            
                            # edited by youyi on 07/10/2020
                            # M_gt_body = relative_pose_cam_to_body(
                            #     delta_Rtijs_4_4_cpu_np[i], Rt_cam2_gt_np[i]
                            # )
                            M_opencv_body = relative_pose_cam_to_body(
                                M_opencv, Rt_cam2_gt_np[i]
                            )
                            dict_of_lists["relative_poses_cam"][base_name].append(
                                # delta_Rtijs_4_4_cpu_np[i]
                                M_opencv
                            )
                            dict_of_lists["relative_poses_body"][base_name].append(
                                M_opencv_body
                                # M_gt_body
                            )
                            # print(f"M_gt_body: {M_gt_body}")
                            ## gt
                            dict_of_lists["err_q"]["gt"].append(error_Rt_gt[0])
                            dict_of_lists["err_t"]["gt"].append(error_Rt_gt[1])
                            dict_of_lists["epi_dists"]["gt"].append(
                                np.expand_dims(epi_dist_mean_gt, -1)
                            )
                            # dict_of_lists["poses"]["gt"].append(delta_Rtijs_4_4_cpu_np[i])
                        else:
                            logging.warning("Failed to recover one of the poses...")
                except KeyboardInterrupt as e:
                    tasks.cancel()
                    raise e

            dict_of_lists["count"] += 1

        # save model
        if (
            self.n_iter % config["training"]["save_interval"] == 0
            and train
            and self.n_iter != 0
        ):
            save_model(
                train_params["save_path"],
                self.net,
                self.n_iter,
                self.n_iter_val,
                self.optimizer,
                loss,
                file_prefix=["deepFNet"],
            )
            # save superpoint network
            if self.if_SP:
                save_model(
                    train_params["save_path"],
                    self.net_SP,
                    self.n_iter,
                    self.n_iter_val,
                    self.optimizer_SP,
                    loss,
                    file_prefix=["superPointNet"],
                )

        return loss.item(), dict_of_lists, self.clamp_cum

    def flush_dict_of_lists(self, writer, task, step, **dict_of_lists):
        config = self.config

        dict_of_lists.pop("count", None)
        for key1 in dict_of_lists.keys():
            for key2 in dict_of_lists[key1].keys():
                dict_of_lists[key1][key2] = np.asarray(
                    dict_of_lists[key1][key2]
                ).flatten()
                print(key1, key2, np.asarray(dict_of_lists[key1][key2]).shape)

        write_metrics_summary(writer, dict_of_lists, task, step)
        epi_dist_mean_est_base = np.stack(
            dict_of_lists["epi_dists"][config["exps"]["base_name"]], axis=0
        ).flatten()
        epi_dist_mean_est_ours = np.stack(
            dict_of_lists["epi_dists"][config["exps"]["our_name"]], axis=0
        ).flatten()
        print(
            "===%s==== Baseline - %s: %.2f, %.2f"
            % (
                task,
                config["exps"]["base_name"],
                np.sum(epi_dist_mean_est_base < 0.1) / epi_dist_mean_est_base.shape[0],
                np.sum(epi_dist_mean_est_base < 1) / epi_dist_mean_est_base.shape[0],
            )
        )
        print(
            "===%s==== Oursssssssssss: %.2f, %.2f"
            % (
                task,
                np.sum(epi_dist_mean_est_ours < 0.1) / epi_dist_mean_est_ours.shape[0],
                np.sum(epi_dist_mean_est_ours < 1) / epi_dist_mean_est_ours.shape[0],
            )
        )
        return init_dict_of_lists(self.config, self.save_lists)
