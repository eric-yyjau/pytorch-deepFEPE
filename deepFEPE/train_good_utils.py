"""
# main training script
Organized and documented by You-Yi on 07/07/2020.

Edition history:
    

Authors:
    You-Yi Jau, Rui Zhu

"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
np.set_printoptions(precision=4, suppress=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cv2
from pebble import ProcessPool
import multiprocessing as mp
from sklearn.metrics import f1_score

ratio_CPU = 0.5
default_number_of_process = int(ratio_CPU * mp.cpu_count())

# our functions
import deepFEPE.dsac_tools.utils_F as utils_F  # If cannot find: export KITTI_UTILS_PATH='/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils'
import deepFEPE.dsac_tools.utils_opencv as utils_opencv  # If cannot find: export KITTI_UTILS_PATH='/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils'
import deepFEPE.dsac_tools.utils_vis as utils_vis  # If cannot find: export KITTI_UTILS_PATH='/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils'
import deepFEPE.dsac_tools.utils_misc as utils_misc  # If cannot find: export KITTI_UTILS_PATH='/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils'
import deepFEPE.dsac_tools.utils_geo as utils_geo  # If cannot find: export KITTI_UTILS_PATH='/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils'
from deepFEPE.models.model_utils import set_nan2zero

# superpoint functions
from superpoint.utils.logging import *
from superpoint.utils.utils import (
    tensor2array,
    save_checkpoint,
    load_checkpoint,
    save_path_formatter,
    flattenDetection
)


def mean_list(list):
    return sum(list) / len(list)

## functions for inference
def mat_E_to_pose(E_ests_layers):
    """ convert from essential matrix to R,t
    params:
        E_ests_layers -> [B, 3, 3]: batch essential matrices
    """
    ## many layers of essential matrices
    for layer_idx, E_ests in enumerate(E_ests_layers):
        R_angle_error_list = []
        t_angle_error_list = []
        t_l2_error_list = []
        q_l2_error_list = []

        # ================= method 1/2 ===============
        ## convert E mat to R, t
        R12s_list = []
        t12s_list = []
        for idx, E_cam in enumerate(E_ests.cpu().transpose(1, 2)):
            # FU, FD, FV= torch.svd(E_cam, some=True)
            # # print('[info.Debug @_E_from_XY] Singular values for recovered E(F):\n', FD.detach().numpy())
            # S_110 = torch.diag(torch.tensor([1., 1., 0.], dtype=FU.dtype, device=FU.device))
            # E_cam = torch.mm(FU, torch.mm(S_110, FV.t()))

            R12s, t12s, M12s = utils_F._get_M2s(E_cam)
            R12s_list.append(R12s)
            t12s_list.append(t12s)
        R12s_batch_cam = [
            torch.stack([R12s[0] for R12s in R12s_list]).to(device),
            torch.stack([R12s[1] for R12s in R12s_list]).to(device),
        ]
        t12s_batch_cam = [
            torch.stack([t12s[0] for t12s in t12s_list]).to(device),
            torch.stack([t12s[1] for t12s in t12s_list]).to(device),
        ]  # already unit norm

    ## aggregate   
    return 0 

##### loss functions #####

def get_loss_softmax(F_gts, pts1_virt, pts2_virt, clamp_at):
    losses = utils_F._sym_epi_dist(
        F_gts, pts1_virt, pts2_virt, if_homo=True, clamp_at=clamp_at
    )
    # print(losses.size(), losses.mean().size())
    # print(losses.detach().cpu().numpy())
    return losses.mean(), losses


def get_Rt_loss(
    E_ests_layers, Ks_cpu, x1_cpu, x2_cpu, delta_Rtijs_4_4_cpu, qs_cam, ts_cam,
    device='cpu'
):
    """ losses from essential matrix and ground truth R,t
    params:
        E_ests_layers -> [B, 3, 3]: batch essential matrices
        Ks_cpu -> [B, 3, 3]: batch intrinsics
        x1_cpu, x2_cpu: no use
        delta_Rtijs_4_4_cpu -> [B, 4, 4]: ground truth transformation matrices
        qs_cam -> [B, 4]: ground truth rotation
        ts_cam -> [B, 3]: ground truth translation

    return:
        dict: with l2 loss, Rt angle error
    
    """
    # Differentiable R t decomposition from E_est
    K_np = Ks_cpu.numpy()
    x1_np, x2_np = x1_cpu.numpy(), x2_cpu.numpy()
    # delta_Rtijs_4_4_cpu_np = delta_Rtijs_4_4_cpu.numpy()

    R_angle_error_layers_list = []
    t_angle_error_layers_list = []
    t_l2_error_layers_list = []
    q_l2_error_layers_list = []
    R_angle_error_mean_layers_list = []
    t_angle_error_mean_layers_list = []
    t_l2_error_mean_layers_list = []
    q_l2_error_mean_layers_list = []

    ## many layers of essential matrices
    for layer_idx, E_ests in enumerate(E_ests_layers):
        R_angle_error_list = []
        t_angle_error_list = []
        t_l2_error_list = []
        q_l2_error_list = []

        # ================= method 1/2 ===============
        ## convert E mat to R, t
        R12s_list = []
        t12s_list = []
        for idx, E_cam in enumerate(E_ests.cpu().transpose(1, 2)):
            # FU, FD, FV= torch.svd(E_cam, some=True)
            # # print('[info.Debug @_E_from_XY] Singular values for recovered E(F):\n', FD.detach().numpy())
            # S_110 = torch.diag(torch.tensor([1., 1., 0.], dtype=FU.dtype, device=FU.device))
            # E_cam = torch.mm(FU, torch.mm(S_110, FV.t()))

            R12s, t12s, M12s = utils_F._get_M2s(E_cam)
            R12s_list.append(R12s)
            t12s_list.append(t12s)
        R12s_batch_cam = [
            torch.stack([R12s[0] for R12s in R12s_list]).to(device),
            torch.stack([R12s[1] for R12s in R12s_list]).to(device),
        ]
        t12s_batch_cam = [
            torch.stack([t12s[0] for t12s in t12s_list]).to(device),
            torch.stack([t12s[1] for t12s in t12s_list]).to(device),
        ]  # already unit norm

        for (
            R1_est_cam,
            R2_est_cam,
            t1_est_cam,
            t2_est_cam,
            q_gt_cam,
            t_gt_cam,
            E_hat_single,
            K_single_np,
            x1_single_np,
            x2_single_np,
            delta_Rtijs_4_4_inv,
        ) in zip(
            R12s_batch_cam[0],
            R12s_batch_cam[1],
            t12s_batch_cam[0],
            t12s_batch_cam[1],
            qs_cam,
            ts_cam,
            E_ests,
            K_np,
            x1_np,
            x2_np,
            torch.inverse(delta_Rtijs_4_4_cpu.to(device)),
        ):
            q1_est_cam = utils_geo._R_to_q(R1_est_cam)
            q2_est_cam = utils_geo._R_to_q(R2_est_cam)
            t_gt_cam = F.normalize(t_gt_cam, p=2, dim=0) # normed translation
            q12_error = [
                utils_geo._l2_error(q1_est_cam, q_gt_cam),
                utils_geo._l2_error(q2_est_cam, q_gt_cam),
            ]
            t12_error = [
                utils_geo._l2_error(t1_est_cam, t_gt_cam),
                utils_geo._l2_error(t2_est_cam, t_gt_cam),
            ]
            q12_who_is_small = q12_error[0] < q12_error[1]
            t12_who_is_small = t12_error[0] < t12_error[1]

            R_est = (
                q12_who_is_small * R1_est_cam + (~q12_who_is_small) * R2_est_cam
            )
            t_est = (
                t12_who_is_small * t1_est_cam + (~t12_who_is_small) * t2_est_cam
            )

            R_gt = delta_Rtijs_4_4_inv[:3, :3]

            # R_angle_error = utils_geo._rot_angle_error(R_est, R_gt)
            R_angle_error = utils_geo.rot12_to_angle_error(
                R_est.detach().cpu().numpy(), R_gt.detach().cpu().numpy()
            )
            t_angle_error = utils_geo.vector_angle(
                t_est.detach().cpu().numpy(), t_gt_cam.detach().cpu().numpy()
            )

            ## calcualte l2 loss
            q_l2_error = (
                q12_who_is_small * q12_error[0]
                + (~q12_who_is_small) * q12_error[1]
            )
            t_l2_error = (
                t12_who_is_small * t12_error[0]
                + (~t12_who_is_small) * t12_error[1]
            )

            # print('--1', layer_idx, R_est.cpu().detach().numpy(), R_gt.cpu().detach().numpy(), R_angle_error)
            # print('--1', layer_idx, R_angle_error, t_angle_error)

            # ================= method 3/2: OpenCV ===============
            # if layer_idx == len(E_ests_layers)-1:
            #     print('---3', E_hat_single)
            #     FU, FD, FV= torch.svd(E_hat_single, some=True)
            #     # print('[info.Debug @_E_from_XY] Singular values for recovered E(F):\n', FD.detach().numpy())
            #     S_110 = torch.diag(torch.tensor([1., 1., 0.], dtype=FU.dtype, device=FU.device))
            #     E_hat_single = torch.mm(FU, torch.mm(S_110, FV.t()))

            #     M_estW, error_Rt_estW, M_estW_cam = utils_F.goodCorr_eval_nondecompose(x1_single_np, x2_single_np, E_hat_single.cpu().detach().numpy().astype(np.float64), delta_Rtijs_4_4_inv.cpu().detach().numpy(), K_single_np, None)
            #     print('--3', M_estW_cam[:3, :3], delta_Rtijs_4_4_inv.cpu().detach().numpy()[:3, :3], error_Rt_estW[0])
            #     # print('--3', layer_idx, error_Rt_estW[0], error_Rt_estW[1])

            # # R2s_batch, t2s_batch = utils_F._get_M2s_batch(E_ests[:2])

            # ================= method 2/2 ===============
            # for E_hat_single, K_single_np, x1_single_np, x2_single_np, delta_Rtijs_4_4_inv, q_gt_cam, t_gt_cam in zip(E_ests, K_np, x1_np, x2_np, torch.inverse(delta_Rtijs_4_4_cpu.cuda()), qs_cam, ts_cam):
            #     M2_list, error_Rt, Rt_cam = utils_F._E_to_M_train(E_hat_single, K_single_np, x1_single_np, x2_single_np, delta_Rt_gt_cam=None, show_debug=False, show_result=False)
            #     if Rt_cam is None:
            #         R_angle_error_list.append(0.)
            #         t_angle_error_list.append(0.)
            #         t_l2_error_list.append(torch.tensor(0.).float().cuda())
            #         q_l2_error_list.append(torch.tensor(0.).float().cuda())
            #         continue

            #     R_est = Rt_cam[:, :3]
            #     # t_est = F.normalize(Rt_cam[:, 3:4], p=2, dim=0)  # already unit norm
            #     t_est = Rt_cam[:, 3:4]
            #     R_gt = delta_Rtijs_4_4_inv[:3, :3]
            #     # t_gt = F.normalize(delta_Rtijs_4_4_inv[:3, 3:4], p=2, dim=0)
            #     t_gt = F.normalize(t_gt_cam, p=2, dim=0)
            #     # t_gt = delta_Rtijs_4_4_inv[:3, 3:4]

            #     R_angle_error = utils_geo._rot_angle_error(R_est, R_gt)
            #     t_angle_error = utils_geo.vector_angle(t_est.detach().cpu().numpy(), t_gt.detach().cpu().numpy())

            #     q_est = utils_geo._R_to_q(R_est)
            #     q_gt = q_gt_cam

            #     q_l2_error = utils_geo._l2_error(q_est, q_gt)
            #     q_l2_error = q_l2_error * (R_angle_error < 30.)
            #     t_l2_error = utils_geo._l2_error(t_est, t_gt)
            #     t_l2_error = t_l2_error * (t_angle_error < 30.)

            R_angle_error_list.append(R_angle_error)
            t_angle_error_list.append(t_angle_error)
            t_l2_error_list.append(t_l2_error)
            q_l2_error_list.append(q_l2_error)

        #     print('--2', layer_idx, R_est.cpu().detach().numpy())

        #     num_inlier, R, t, mask_new = cv2.recoverPose(E_hat_single.cpu().detach().numpy().astype(np.float64), x1_single_np, x2_single_np, focal=K_single_np[0, 0], pp=(K_single_np[0, 2], K_single_np[1, 2]))
        #     print('--3', layer_idx, R)

        # ================================

        # if layer_idx == len(E_ests_layers)-1:
        #     print('R_angle_error_list', R_angle_error_list)
        #     print('t_angle_error_list', t_angle_error_list)

        R_angle_error_mean = sum(R_angle_error_list) / len(R_angle_error_list)
        t_angle_error_mean = sum(t_angle_error_list) / len(t_angle_error_list)
        t_l2_error_mean = sum(t_l2_error_list) / len(t_l2_error_list)
        q_l2_error_mean = sum(q_l2_error_list) / len(q_l2_error_list)

        R_angle_error_layers_list.append(np.array(R_angle_error_list))
        t_angle_error_layers_list.append(np.array(t_angle_error_list))
        t_l2_error_layers_list.append(torch.stack(t_l2_error_list))
        q_l2_error_layers_list.append(torch.stack(q_l2_error_list))

        R_angle_error_mean_layers_list.append(R_angle_error_mean)
        t_angle_error_mean_layers_list.append(t_angle_error_mean)
        t_l2_error_mean_layers_list.append(t_l2_error_mean)
        q_l2_error_mean_layers_list.append(q_l2_error_mean)

    R_angle_error_mean_all = mean_list(R_angle_error_mean_layers_list)
    t_angle_error_mean_all = mean_list(t_angle_error_mean_layers_list)
    t_l2_error_mean_all = mean_list(t_l2_error_mean_layers_list)
    q_l2_error_mean_all = mean_list(q_l2_error_mean_layers_list)

    return_list = {
        "t_l2_error_mean": t_l2_error_mean_all,
        "q_l2_error_mean": q_l2_error_mean_all,
        "t_l2_error_list": torch.stack(t_l2_error_mean_layers_list),
        "q_l2_error_list": torch.stack(t_l2_error_mean_layers_list),
    }
    return_list.update(
        {
            "R_angle_error_mean": R_angle_error_mean_all,
            "R_angle_error_list": np.array(R_angle_error_mean_layers_list),
            "t_angle_error_mean": t_angle_error_mean_all,
            "t_angle_error_list": np.array(t_angle_error_mean_layers_list),
        }
    )
    return_list.update(
        {
            "R_angle_error_layers_list": R_angle_error_layers_list,
            "t_angle_error_layers_list": t_angle_error_layers_list,
            "t_l2_error_layers_list": t_l2_error_layers_list,
            "q_l2_error_layers_list": q_l2_error_layers_list,
        }
    )

    return return_list



def get_all_loss_DeepF(
    outs, pts1_virt_ori, pts2_virt_ori, Ks, loss_params, get_residual_summaries=True
):
    """ get f-loss
    params:
        outs: output from deepF,
        pts1_virt_ori, pts2_virt_ori: virtual points 
        Ks: intrinsics

    return:
        dict: with residuals
    """
    # if loss_params['if_img_feat']:
    logits_softmax = outs["weights"]
    # else:
    #     logits = outs['logits'] # [batch_size, N]
    #     logits_softmax = F.softmax(logits, dim=1)
    # loss_E = 0.

    F_est_normalized, T1, T2, out_layers, residual_layers, weights_layers = (
        outs["F_est"],
        outs["T1"],
        outs["T2"],
        outs["out_layers"],
        outs["residual_layers"],
        outs["weights_layers"],
    )
    pts1_eval = (T1 @ pts1_virt_ori.permute(0, 2, 1)).permute(0, 2, 1)
    pts2_eval = (T2 @ pts2_virt_ori.permute(0, 2, 1)).permute(0, 2, 1)

    # pts1_eval = utils_misc._homo(F.normalize(pts1_eval[:, :, :2], dim=2))
    # pts2_eval = utils_misc._homo(F.normalize(pts2_eval[:, :, :2], dim=2))

    loss_layers = []
    losses_layers = []
    E_ests_layers = []
    loss_min_layers = []
    loss_min_batch = []

    loss_F_all = 0.0
    for iter in range(loss_params["depth"]):
        # logging.info(f"fund. {iter}:{out_layers[iter]}")
        losses = utils_F.compute_epi_residual(
            pts1_eval, pts2_eval, out_layers[iter], loss_params["clamp_at"]
        )
        # logging.info(f"losses. {iter}:{losses.shape}")
        # if losses.max() == loss_params["clamp_at"]: logging.info(f"clamp_at: {loss_params['clamp_at']}")
        # losses = utils_F._YFX(pts1_eval, pts2_eval, out_a[iter], if_homo=True, clamp_at=loss_params['clamp_at'])
        # loss_min_layers.append(losses.mean(dim=1))
        loss_min_batch.append(losses.mean(dim=1))

        losses_layers.append(losses)
        # if iter==loss_params['depth']-1:
        #     print(losses[:, :10].detach().cpu().numpy())
        loss = losses.mean()
        loss_layers.append(loss)
        loss_F_all += loss

        E_ests_layers.append(
            Ks.transpose(1, 2) @ T2.permute(0, 2, 1) @ out_layers[iter] @ T1 @ Ks
        )

    loss_min_batch = torch.stack(loss_min_batch)
    # logging.info(f"loss_min_batch. :{loss_min_batch.shape}")


    loss_F_all = loss_F_all / len(loss_layers)

    F_ests = (
        T2.permute(0, 2, 1) @ F_est_normalized @ T1
    )  # If use norm_K, then the output F_est is esentially E_ests, and the next line basically transforms it back: E_ests == Ks.transpose(1, 2) @ {T2.permute(0,2,1).bmm(F_est.bmm(T1))} @ Ks
    E_ests = Ks.transpose(1, 2) @ F_ests @ Ks

    ## loss dictionary
    losses_dict = {
        "loss_layers": loss_layers,
        "loss_F": loss_F_all,
        "loss_min_layers": loss_min_batch.min(dim=1)[0],  # np[layer, batch]. [0] to get the value
        "loss_min_batch":  loss_min_batch.min(dim=0)[0],
    }

    if loss_params["if_tri_depth"]:
        depths_mean = outs["depths_mean"]
        tri_depths = outs["tri_depths"]
        loss_depth = torch.abs(1.0 / depths_mean - 1.0 / tri_depths).clamp(0.0, 2.0)
        loss_depth_mean = loss_depth.mean()
        losses_dict.update({"loss_depth_mean": loss_depth_mean})
        # print(loss_depth[0][0][:100].detach().cpu().numpy())

    if loss_params["if_sample_loss"]:
        out_sample_selected_batch_layers, weights_sample_selected_accu_batch_layers = (
            outs["out_sample_selected_batch_layers"],
            outs["weights_sample_selected_accu_batch_layers"],
        )
        loss_selected_layers = []
        for layer_idx in range(loss_params["depth"]):
            loss_selected_batch = []
            # print(weights_sample_selected_accu_batch_layers[layer_idx], torch.sum(weights_sample_selected_accu_batch_layers[layer_idx]), weights_sample_selected_accu_batch_layers[layer_idx].size())
            for selected_idx, (selected_F, select_w) in enumerate(
                zip(
                    out_sample_selected_batch_layers[layer_idx],
                    weights_sample_selected_accu_batch_layers[layer_idx],
                )
            ):
                loss_select_F = utils_F.compute_epi_residual(
                    pts1_eval[selected_idx].unsqueeze(0),
                    pts2_eval[selected_idx].unsqueeze(0),
                    selected_F,
                    0.02,
                )  # [selects_each_sample, 100(num of virt corres)]
                # loss_params['clamp_at'])
                # print(loss_select_F.size(), select_w.size(), torch.sum(select_w))
                loss_selected_batch.append(loss_select_F.mean())
                # print(loss_select_F)
                # loss_select_F = loss_select_F * select_w # [selects_each_sample, 100(num of virt corres)], [selects_each_sample, 1]
                # loss_selected_batch.append(torch.sum(loss_select_F, dim=0).mean())

            loss_selected_layers.append(
                sum(loss_selected_batch) / len(loss_selected_batch)
            )

        loss_selected_all = sum(loss_selected_layers) / len(loss_selected_layers)

        losses_dict.update(
            {
                "loss_selected_F": loss_selected_all,
                "loss_selected_layers": loss_selected_layers,
            }
        )

    # print(len(outs['epi_res_layers']), len(outs['weights_layers']))
    loss_epi_res_all = 0.0
    loss_epi_res_layers = []
    if loss_params["depth"] > 1:
        for epi_res, weights in zip(outs["epi_res_layers"], outs["weights_layers"]):
            epi_res_weighted = epi_res * weights
            loss_epi_res_layers.append(epi_res_weighted.mean())
        loss_epi_res_all = sum(loss_epi_res_layers) / len(loss_epi_res_layers)
    losses_dict.update(
        {"loss_epi_res_layers": loss_epi_res_layers, "loss_epi_res": loss_epi_res_all}
    )

    residual_norm_layers, residual_norm_max_layers = None, None
    if get_residual_summaries:
        topK = loss_params["topK"]
        matches_good_unique_nums = loss_params[
            "matches_good_unique_nums"
        ]  # .numpy().tolist()
        ## Residual loss
        residual_norm_layers = []
        residual_norm_topK_layers = []
        residual_norm_max_layers = []
        for residual in residual_layers:  # [B, N]
            residual_norms = residual.norm(p=2, dim=1)
            residual_norm_layers.append(residual_norms.mean())
            residual_norm_max_layers.append(residual_norms.max())
            residual_norms_topK_unique = get_unique(
                residual, topK, matches_good_unique_nums
            )  # [B, topk]
            residual_norm_topK_layers.append(residual_norms_topK_unique.mean())

        loss_residual_all = sum(residual_norm_layers) / len(residual_norm_layers)
        loss_residual_topK_all = sum(residual_norm_topK_layers) / len(
            residual_norm_topK_layers
        )

        ## Regularize by weight thres
        loss_regW_clip_layers = []
        # regW_thres = 1./8.
        regW_thres = 0.01
        for weights_layer in weights_layers:  # [B, 1, N]
            loss_regW_thres = nn.ReLU(inplace=True)(weights_layer - regW_thres).mean()
            loss_regW_clip_layers.append(loss_regW_thres)
        loss_regW_clip_all = sum(loss_regW_clip_layers) / len(weights_layers) * 100.0

        ## Top K weights / all weights entropy regularizer
        loss_regW_entro_layers = []
        loss_regW_entro_topK_layers = []
        for weights_layer in weights_layers:  # [B, 1, N]
            weights_entropy = (
                torch.distributions.Categorical(probs=weights_layer.squeeze())
                .entropy()
                .mean()
            )
            loss_regW_entro_layers.append(weights_entropy)

            weights_layer_topK_unique = get_unique(
                weights_layer.squeeze(1), topK, matches_good_unique_nums
            )  # [B, topk]
            # weights_layer_topK_unique = torch.topk(weights_layer.squeeze(), topK, dim=1)[0]
            weights_entropy_topK = (
                torch.distributions.Categorical(
                    probs=weights_layer_topK_unique.squeeze()
                )
                .entropy()
                .mean()
            )  # max: 2.0794
            loss_regW_entro_topK_layers.append(weights_entropy_topK)

        loss_regW_entro_all = sum(loss_regW_entro_layers) / len(weights_layers)
        loss_regW_entro_topK_all = sum(loss_regW_entro_topK_layers) / len(
            weights_layers
        )

        losses_dict.update(
            {
                "loss_residual": loss_residual_all,
                "loss_residual_topK": loss_residual_topK_all,
                "loss_regW_clip": loss_regW_clip_all,
                "loss_regW_entro": loss_regW_entro_all,
                "loss_regW_entro_topK": loss_regW_entro_topK_all,
            }
        )

    return (
        losses_dict,
        E_ests,
        F_ests,
        logits_softmax,
        residual_norm_layers,
        residual_norm_max_layers,
        E_ests_layers,
    )



##### end loss functions #####

def get_E_ests(x1, x2, Ks, logits_weights, if_normzliedK=True):
    # E_ests_list = []
    # for x1_single, x2_single, K, w in zip(x1, x2, Ks, logits_weights):
    #     E_est = utils_F._E_from_XY(x1_single, x2_single, K, torch.diag(w), if_normzliedK=if_normzliedK)
    #     E_ests_list.append(E_est)
    # E_ests = torch.stack(E_ests_list)
    E_ests = utils_F._E_from_XY_batch(
        x1,
        x2,
        Ks,
        torch.diag_embed(logits_weights, dim1=-2, dim2=-1),
        if_normzliedK=if_normzliedK,
    )
    return E_ests



def get_unique(xs, topk, matches_good_unique_nums):  # [B, N]
    xs_topk_list = []
    for x, matches_good_unique_num in zip(xs, matches_good_unique_nums):
        # x_unique = torch.unique(x) # no gradients!!!
        x_unique = x[:matches_good_unique_num]
        x_unique_topK = torch.topk(x_unique, topk)[0]
        xs_topk_list.append(x_unique_topK)
    return torch.stack(xs_topk_list)

##### validation #####
def val_rt(
    idx,
    K_np,
    x1_single_np,
    x2_single_np,
    E_est_np,
    E_gt_np,
    F_est_np,
    F_gt_np,
    delta_Rtijs_4_4_cpu_np,
    five_point,
    if_opencv=True,
):
    """ from essential matrix, get Rt, and error
    params:
        K_np:
        x1_single_np: matching point x1
        x2_single_np: matching point x2
        E_est_np, E_gt_np: essential matrix
        F_est_np, F_gt_np: fundamental matrix
        delta_Rtijs_4_4_cpu_np: ground truth transformation matrix
        five_point: with five_point or not (default not)
        if_opencv: compare to the results using opencv
    return:
        (pose error), (epipolar distance), (reconstructed poses)
    """
    delta_Rtij_inv = np.linalg.inv(delta_Rtijs_4_4_cpu_np)[:3]

    error_Rt_estW = None
    epi_dist_mean_estW = None
    error_Rt_opencv = None
    epi_dist_mean_opencv = None

    # Evaluating with our weights
    # _, error_Rt_estW = utils_F._E_to_M(E_est.detach(), K, x1_single_np, x2_single_np, w>0.5, \
    #     delta_Rtij_inv, depth_thres=500., show_debug=False, show_result=False, method_name='Est ws')
    M_estW, error_Rt_estW = utils_F.goodCorr_eval_nondecompose(
        x1_single_np,
        x2_single_np,
        E_est_np.astype(np.float64),
        delta_Rtij_inv,
        K_np,
        None,
    )
    M_gt, error_Rt_gt = utils_F.goodCorr_eval_nondecompose(
        x1_single_np,
        x2_single_np,
        E_gt_np.astype(np.float64),
        delta_Rtij_inv,
        K_np,
        None,
    )
    epi_dist_mean_estW, _, _ = utils_F.epi_distance_np(
        F_est_np, x1_single_np, x2_single_np, if_homo=False
    )
    epi_dist_mean_gt, _, _ = utils_F.epi_distance_np(
        F_gt_np, x1_single_np, x2_single_np, if_homo=False
    )

    # print('-0', F_est_np, epi_dist_mean_estW)

    # Evaluating with OpenCV 5-point
    if if_opencv:
        M_opencv, error_Rt_opencv, _, E_return = utils_opencv.recover_camera_opencv(
            K_np,
            x1_single_np,
            x2_single_np,
            delta_Rtij_inv,
            five_point=five_point,
            threshold=0.01,
            show_result=False,
        )
        if five_point:
            E_recover_opencv = E_return
            F_recover_opencv = utils_F.E_to_F_np(E_recover_opencv, K_np)
        else:
            E_recover_opencv, F_recover_opencv = E_return[0], E_return[1]
        # print('+++', K_np)
        epi_dist_mean_opencv, _, _ = utils_F.epi_distance_np(
            F_recover_opencv, x1_single_np, x2_single_np, if_homo=False
        )
        # print('-0-', utils_F.E_to_F_np(E_recover_5point, K_np))
        # print('-1', utils_F.E_to_F_np(E_recover_5point, K_np), epi_dist_mean_5point)
    return (
        error_Rt_estW, # error R,t
        epi_dist_mean_estW, # epipolar distance for each corr
        error_Rt_opencv,
        epi_dist_mean_opencv,
        error_Rt_gt, # for sanity check
        epi_dist_mean_gt, # epipolar distance wrt gt
        idx,
        M_estW, # reconstructed R, t
        M_opencv,
    )

##### superpoint helping functions
def get_matches_from_SP(imgs_grey, net_SP, SP_processer, SP_tracker, out_num_points=1000):
    """ get matching from superpoint
    params: 
        imgs_grey: images
        net_SP: superpoint network
        SP_processer: batch NMS
        SP_tracker: nn match

    return:
        xs: sampled SP points
        offset: sample SP residuals
        xs_SP: unique SP points with residuals
    """
    from train_good_utils import process_SP_output
    imgs_grey_float = [img_grey.float().cuda() / 255.0 for img_grey in imgs_grey]
    f = lambda x: x.cpu().detach().numpy()

    xs_SP = []
    deses_SP = []
    reses_SP = []
    for idx, img12_grey_float in enumerate(imgs_grey_float):
        outs = net_SP(
            img12_grey_float.unsqueeze(-1).permute(0, 3, 1, 2)
        )  # [batch_size, 1, H, W]
        # outs = net_SP.process_output(outs, SP_processer)
        outs = process_SP_output(outs, SP_processer)
        xs_SP.append(outs["pts_int"])
        deses_SP.append(outs["pts_desc"])
        reses_SP.append(outs["pts_offset"])

    batch_size = imgs_grey[0].size(0)

    xs_list = []
    offsets_list = []
    quality_list = []
    num_matches_list = []
    for batch_idx in range(batch_size):
        matching_mask = SP_tracker.nn_match_two_way(
            f(deses_SP[0][batch_idx]).transpose(),
            f(deses_SP[1][batch_idx]).transpose(),
            nn_thresh=SP_tracker.nn_thresh,
        )  # [3, N_i]
        print(f"nn_thresh = {SP_tracker.nn_thresh}, matches: {matching_mask.shape}")
        choice = utils_misc.crop_or_pad_choice(
            matching_mask.shape[1], out_num_points=out_num_points, shuffle=True
        )
        num_matches_list.append(matching_mask.shape[1])
        matching_mask = matching_mask[:, choice]
        pts_m = []
        pts_m_res = []
        for i in range(2):
            x = xs_SP[i][batch_idx][matching_mask[i, :].astype(int), :]
            res = reses_SP[i][batch_idx][matching_mask[i, :].astype(int), :]
            pts_m.append(x)
            pts_m_res.append(res)
            pass

        pts_m = torch.cat((pts_m[0], pts_m[1]), dim=1)
        xs_list.append(pts_m)
        #     matches_test = toNumpy(pts_m)

        pts_m_res = torch.cat((pts_m_res[0], pts_m_res[1]), dim=1)
        offsets_list.append(pts_m_res)

        quality_list.append(matching_mask[2:3, :].transpose())

        # utils_vis.draw_corr(sample['imgs'][0][batch_idx], sample['imgs'][1][batch_idx], matches_test[:, :2], matches_test[:, 2:], linewidth=2., title='Sample of 100 corres.')

    xs = torch.stack(xs_list)
    offsets = torch.stack(offsets_list)
    num_matches = torch.tensor(num_matches_list)
    quality = torch.from_numpy(np.stack(quality_list)).cuda().float()

    xs_all = [x+r for (x,r) in zip(xs_SP, reses_SP)]
    # return xs, offsets, quality
    return {'xs': xs, 'offsets': offsets, 'quality': quality, 'num_matches': num_matches, 'xs_SP': xs_all}


def process_SP_output(output, sp_processer):
    # self.print_p()
    """ # superpoint get sparse points and residuals
    input:
        N: number of points
    return: -- type: tensorFloat
        pts: tensor [batch, N, 2] (no grad)  (x, y)
        pts_offset: tensor [batch, N, 2] (grad) (x, y)
        pts_desc: tensor [batch, N, 256] (grad)
    """
    # from models.model_utils import pred_soft_argmax, sample_desc_from_points
    semi = output['semi']
    desc = output['desc']
    # flatten
    semi = set_nan2zero(semi)
    desc = set_nan2zero(desc)
    heatmap = flattenDetection(semi) # [batch_size, 1, H, W]
    # nms
    heatmap_nms_batch = sp_processer.heatmap_to_nms(heatmap, tensor=True)
    # extract offsets
    outs = sp_processer.pred_soft_argmax(heatmap_nms_batch, heatmap)
    residual = outs['pred']
    # extract points
    outs = sp_processer.batch_extract_features(desc, heatmap_nms_batch, residual)

    # output.update({'heatmap': heatmap, 'heatmap_nms': heatmap_nms, 'descriptors': descriptors})
    output.update(outs)
    # self.output = output
    return output
##### end functions #####

def write_metrics_summary(writer, dict_of_lists, task, n_iter):
    metric_list = list(dict_of_lists.keys())
    exp_list = list(dict_of_lists[metric_list[0]].keys())
    # print(exp_list)
    # exp_list.remove('gt')
    # print(exp_list)
    assert "epi_dists" in metric_list
    epi_dists_list_gt = dict_of_lists["epi_dists"]["gt"]
    epi_dists_gt = np.stack(epi_dists_list_gt, axis=0).flatten()
    pred_01_true = epi_dists_gt < 0.1
    pred_1_true = epi_dists_gt < 1.0

    for _tag_exp in exp_list:
        # epi_dist ratios with ths of 0.1 and 1.
        epi_dists_list = dict_of_lists["epi_dists"][_tag_exp]
        epi_dists = np.stack(epi_dists_list, axis=0).flatten()
        ratio_01 = np.sum(epi_dists < 0.1) / np.shape(epi_dists)[0]
        ratio_1 = np.sum(epi_dists < 1.0) / np.shape(epi_dists)[0]
        writer.add_scalar(
            task + "-Error-epi_dists/%s-0.1" % (_tag_exp), ratio_01, n_iter
        )
        writer.add_scalar(task + "-Error-epi_dists/%s-1" % (_tag_exp), ratio_1, n_iter)
        logging.info(
            "-- %s: ratio 0.1: %.2f; ratio 1.: %.2f" % (_tag_exp, ratio_01, ratio_1)
        )

        # F-1 scores with epi_dist ths of 0.1 and 1.
        # https://skymind.ai/wiki/accuracy-precision-recall-f1
        writer.add_scalar(
            task + "-Error-F1/%s-0.1" % (_tag_exp),
            f1_score(pred_01_true, epi_dists < 0.1),
            n_iter,
        )
        writer.add_scalar(
            task + "-Error-F1/%s-1" % (_tag_exp),
            f1_score(pred_1_true, epi_dists < 1.0),
            n_iter,
        )

        for _sub_tag_metric in metric_list:
            if _sub_tag_metric != "epi_dists":
                writer.add_scalar(
                    task + "-Error-Median/%s-%s" % (_sub_tag_metric, _tag_exp),
                    np.median(dict_of_lists[_sub_tag_metric][_tag_exp]),
                    n_iter,
                )

        # angular errors and AUC of rotation(q) and translation(t) with multiple thses
        ths = [0.0, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 90.0, 180.0]
        cur_err_q = np.array(dict_of_lists["err_q"][_tag_exp]).flatten()
        cur_err_t = np.array(dict_of_lists["err_t"][_tag_exp]).flatten()
        writer.add_scalar(
            task + "-Error-MAX/err_q_MAX_%s" % _tag_exp, np.amax(cur_err_q), n_iter
        )
        writer.add_scalar(
            task + "-Error-MAX/err_t_MAX_%s" % _tag_exp, np.amax(cur_err_t), n_iter
        )
        writer.add_histogram(
            task + "-Error-hist/err_q_%s" % _tag_exp, cur_err_q, n_iter
        )
        writer.add_histogram(
            task + "-Error-hist/err_t_%s" % _tag_exp, cur_err_t, n_iter
        )
        writer.add_histogram(
            task + "-Error-hist/err_t-Clip10.degree_%s" % _tag_exp,
            np.clip(cur_err_t, 0.0, 10.0),
            n_iter,
        )
        # Get histogram
        q_acc_hist, _ = np.histogram(cur_err_q, ths)
        t_acc_hist, _ = np.histogram(cur_err_t, ths)
        # qt_acc_hist, _ = np.histogram(np.maximum(cur_err_q, cur_err_t), ths)
        num_pair = float(cur_err_q.shape[0])
        q_acc_hist = q_acc_hist.astype(float) / num_pair
        t_acc_hist = t_acc_hist.astype(float) / num_pair
        # qt_acc_hist = qt_acc_hist.astype(float) / num_pair
        q_acc = np.cumsum(q_acc_hist)
        t_acc = np.cumsum(t_acc_hist)
        # qt_acc = np.cumsum(qt_acc_hist)
        # Store return val
        # if _tag_exp=='gt':
        #     print(cur_err_q)
        #     print(q_acc_his[t)
        #     print(q_acc)
        for _idx_th in range(1, len(ths)):
            # writer.add_scalar(task+'-Error-AUC/acc_q_auc{}_{}'.format(ths[_idx_th], _tag_exp), np.mean(q_acc[:_idx_th]), n_iter)
            # writer.add_scalar(task+'-Error-AUC/acc_t_auc{}_{}'.format(ths[_idx_th], _tag_exp), np.mean(t_acc[:_idx_th]), n_iter)
            # writer.add_scalar(task+'-Error-AUC/acc_qt_auc{}_{}'.format(ths[_idx_th], _tag_exp), np.mean(qt_acc[:_idx_th]), n_iter)
            writer.add_scalar(
                task + "-Error-ratio/ratio_q{}_{}".format(ths[_idx_th], _tag_exp),
                q_acc[_idx_th - 1],
                n_iter,
            )
            writer.add_scalar(
                task + "-Error-ratio/ratio_t{}_{}".format(ths[_idx_th], _tag_exp),
                t_acc[_idx_th - 1],
                n_iter,
            )
            # writer.add_scalar(task+'-Error-ratio/ratio_qt{}_{}'.format(ths[_idx_th], _tag_exp), qt_acc[_idx_th-1], n_iter)


def get_mean_std(dict_of_lists, if_print=False):
    dict_of_results = {}
    for key in dict_of_lists.keys():
        errors = dict_of_lists[key]
        errors_R = [error[0] for error in errors]
        errors_t = [error[1] for error in errors]
        results = [
            [np.mean(errors_R), np.std(errors_R), np.median(errors_R)],
            [np.mean(errors_t), np.std(errors_t), np.median(errors_t)],
        ]
        if if_print:
            print(np.sort(errors_R)[::-1])

            print(
                "[%s - %d samples] " % (key, len(errors_R))
                + "Error_R = "
                + toCyan("%.4f ± %.4f" % (results[0][0], results[0][1]))
                + ", med "
                + toRed("%.4f" % results[0][2])
                + "; Error_t = "
                + toCyan("%.4f ± %.4f" % (results[1][0], results[1][1]))
                + ", med "
                + toRed("%.4f" % results[1][2])
            )
        dict_of_results.update({key: results})

    return dict_of_results


def toNumpy(tensor):
    return tensor.detach().cpu().numpy()


##### model helper #####
def save_model(save_path, net, n_iter, n_iter_val, optimizer, loss):
    model_state_dict = net.module.state_dict()
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
    )
    logging.info("save model at training step: %d", n_iter)


def prepare_model(config, net, device, n_iter, n_iter_val, net_postfix=""):
    """ load pretrained weights, iterations, and optimizer
    params:
        net: torch network
        device: 'cpu' or 'cuda'
        net_postfix: specified in config file
        n_iter, n_iter_val: can be overwritten
    return:
        net, optimizer: suppose in the gpu
        n_iter, n_iter_val
    """
    
    # load network
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
    net = nn.DataParallel(net)

    # optimizer
    logging.info("+++[Train]+++ setting adam solver")
    optimizer = optim.Adam(net.parameters(), lr=config["training"]["learning_rate"])

    if config["training"]["retrain" + net_postfix] == False:
        logging.info(
            "Loading optimizer: path: %s, mode: %s" % (checkpoint_path, checkpoint_mode)
        )
        optimizer, n_iter, n_iter_val = pretrainedLoader_opt(
            optimizer, n_iter, checkpoint_path, mode=checkpoint_mode, full_path=True
        )

    # reset iterations
    if config["training"]["reset_iter" + net_postfix]:
        logging.info("reset iterations to 0")
        n_iter = 0
    logging.info("n_iter starts at %d" % n_iter)

    return net, optimizer, n_iter, n_iter_val

##### end functions #####


##### deprecated #####
# def get_all_loss(
#     outs,
#     x1_normalizedK,
#     x2_normalizedK,
#     pts1_virt_normalizedK,
#     pts2_virt_normalizedK,
#     Ks,
#     E_gts,
#     loss_params,
# ):
#     logits = outs["logits"]  # [batch_size, N]
#     logits_softmax = F.softmax(logits, dim=1)

#     E_ests = None
#     loss_E = 0.0
#     E_ests = get_E_ests(
#         x1_normalizedK, x2_normalizedK, Ks, logits_softmax, if_normzliedK=True
#     )

#     loss_all, losses = get_loss_softmax(
#         E_ests, pts1_virt_normalizedK, pts2_virt_normalizedK, loss_params["clamp_at"]
#     )
#     losses_np = losses.detach().cpu().numpy()
#     if np.isnan(np.amax(losses_np)):
#         programPause = raw_input("Press the <ENTER> key to continue...")

#     loss_layers = []
#     loss_layers.append(loss_all)

#     if loss_params["model"] == "GoodCorresNet_layers":
#         # logging.info('Adding losses from following layers...')
#         for E_ests in outs["E_ests_list"]:
#             loss, losses = get_loss_softmax(
#                 E_ests,
#                 pts1_virt_normalizedK,
#                 pts2_virt_normalizedK,
#                 loss_params["clamp_at"],
#             )
#             loss_layers.append(loss)
#             loss_all += loss
#         loss_all = loss_all / len(loss_layers)

#     return loss_all, loss_layers, loss_E, E_ests, logits_softmax, logits

