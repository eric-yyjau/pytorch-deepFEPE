""" top level of deepF network, refer to ErrorEstimator for network structure
Reviewed and tested by You-Yi on 07/13/2020.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch.nn.functional import grid_sample
import numpy as np
import cv2

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# import deepFEPE.dsac_tools.utils_F as utils_F # If cannot find: export KITTI_UTILS_PATH='/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils'
from deepFEPE.dsac_tools import utils_F as utils_F # If cannot find: export KITTI_UTILS_PATH='/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils'
# import utils_F.compute_epi_residual as compute_epi_residual
# import utils_F.compute_epi_residual_non_rob as compute_epi_residual_non_rob

from deepFEPE.models.GoodCorresNet import GoodCorresNet
from deepFEPE.models.ErrorEstimators import ErrorEstimator
from deepFEPE.models.model_utils import set_nan2zero
import logging

#### class specific


class NormalizeAndExpand(nn.Module):
    def __init__(self, is_cuda=True, is_test=False):
        super(NormalizeAndExpand, self).__init__()

        self.ones_b = Variable(torch.ones((1, 1, 1)), volatile=is_test)
        self.T_b = Variable(torch.zeros(1, 3, 3), volatile=is_test)

        if is_cuda:
            self.ones_b = self.ones_b.cuda()
            self.T_b = self.T_b.cuda()

    def normalize(self, pts):
        T = self.T_b.expand(pts.size(0), 3, 3).clone()
        ones = self.ones_b.expand(pts.size(0), pts.size(1), 1)

        pts = torch.cat((pts, ones), 2)

        c = torch.mean(pts,1)
        newpts_ = (pts - c.unsqueeze(1)) # First center to zero mean
        meandist = newpts_[:,:,:2].pow(2).sum(2).sqrt().mean(1)

        scale = 1.0/meandist

        T[:,0,0] = scale
        T[:,1,1] = scale
        T[:,2,2] = 1
        T[:,0,2] = -c[:,0]*scale
        T[:,1,2] = -c[:,1]*scale

        pts_out = torch.bmm(T, pts.permute(0,2,1))
        return pts_out, T

    def forward(self, pts):
        pts1, T1 = self.normalize(pts[:,:,:2])
        pts2, T2 = self.normalize(pts[:,:,2:])


        return pts1, pts2, T1, T2

class NormalizeAndExpand_K(nn.Module):
    def __init__(self, is_cuda=True, is_test=False):
        super(NormalizeAndExpand_K, self).__init__()

        self.ones_b = Variable(torch.ones((1, 1, 1)), volatile=is_test)
        self.T_b = Variable(torch.zeros(1, 3, 3), volatile=is_test)

        if is_cuda:
            self.ones_b = self.ones_b.cuda()
            self.T_b = self.T_b.cuda()

    def normalize(self, pts, K_invs):
        T = K_invs
        ones = self.ones_b.expand(pts.size(0), pts.size(1), 1)
        pts = torch.cat((pts, ones), 2)
        pts_out = torch.bmm(T, pts.permute(0,2,1))
        return pts_out, T

    def forward(self, pts, K_invs):
        pts1, T1 = self.normalize(pts[:,:,:2], K_invs)
        pts2, T2 = self.normalize(pts[:,:,2:], K_invs)

        return pts1, pts2, T1, T2

class NormalizeAndExpand_HW(nn.Module):
    # so that the coordintes are normalized to [-1, 1] for both H and W
    def __init__(self, image_size, is_cuda=True, is_test=False):
        super(NormalizeAndExpand_HW, self).__init__()

        self.ones_b = Variable(torch.ones((1, 1, 1)), volatile=is_test)
        # self.T_b = Variable(torch.zeros(1, 3, 3), volatile=is_test)

        self.H, self.W = image_size[0], image_size[1]

        if is_cuda:
            self.ones_b = self.ones_b.cuda()
            # self.T_b = self.T_b.cuda()
            # self.T = self.T.cuda()

    def normalize(self, pts):

        ones = self.ones_b.expand(pts.size(0), pts.size(1), 1)
        T = torch.tensor([[2./self.W, 0., -1.], [0., 2./self.H, -1.], [0., 0., 1.]], device=pts.device, dtype=pts.dtype).unsqueeze(0).expand(pts.size(0), -1, -1)
        pts = torch.cat((pts, ones), 2)
        pts_out = T @ pts.permute(0,2,1)
        return pts_out, T

    def forward(self, pts):
        pts1, T1 = self.normalize(pts[:,:,:2])
        pts2, T2 = self.normalize(pts[:,:,2:])

        return pts1, pts2, T1, T2

## get fundamental matrix and residuals
class Fit(nn.Module):
    def __init__(self, is_cuda=True, is_test=False, if_cpu_svd=False, normalize_SVD=True):
        super(Fit, self).__init__()
        # self.svd = bsvd(is_cuda, is_test)

        self.ones_b = Variable(torch.ones((1, 1, 1)).float())
        self.zero_b = Variable(torch.zeros((1, 1, 1)).float())
        self.T_b = torch.zeros(1, 3, 3).float()

        self.mask = Variable(torch.ones(3))
        self.mask[-1] = 0

        self.normalize_SVD = normalize_SVD
        self.if_cpu_svd = if_cpu_svd
        if self.if_cpu_svd:
            self.mask_cpu = self.mask.clone()

        if is_cuda:
            self.ones_b = self.ones_b.cuda()
            self.zero_b = self.zero_b.cuda()
            self.T_b = self.T_b.cuda()
            self.mask = self.mask.cuda()
        self.is_cuda = is_cuda
        # self.bsvd = bsvd_torch()

    def normalize(self, pts, weights):
        """ normalize the points to the weighted center

        """
        device = pts.device
        T = Variable(self.T_b.to(device).expand(pts.size(0), 3, 3)).clone()
        ones = self.ones_b.to(device).expand(pts.size(0), pts.size(1), 1)

        denom = weights.sum(1)

        # c = torch.mean(pts,1)
        # newpts_ = (pts - c.unsqueeze(1))
        # meandist = newpts_[:,:,:2].pow(2).sum(2).sqrt().mean(1)

        ## get the center
        c = torch.sum(pts*weights,1)/denom
        # print(c.size(), pts.size())
        newpts_ = (pts - c.unsqueeze(1))
        meandist = ((weights*(newpts_[:,:,:2].pow(2).sum(2).sqrt().unsqueeze(2))).sum(1)/denom).squeeze(1)

        scale = 1.4142/meandist

        T[:,0,0] = scale
        T[:,1,1] = scale
        T[:,2,2] = 1
        T[:,0,2] = -c[:,0]*scale
        T[:,1,2] = -c[:,1]*scale

        # pts_ = torch.cat((pts, ones), 2)
        # print(pts.device, weights.device, T.device, self.T_b.device)
        pts_out = torch.bmm(T, pts.permute(0,2,1))
        return pts_out, T

    def weighted_svd(self, pts1, pts2, weights, if_print=False):
        """ main function: get fundamental matrix and residual
        params: 
            pts1 -> [B, N, 2]: first set of points
            pts2 -> [B, N, 2]: second set of points
            weights -> [B, N, 1]: predicted weights
        return:
            out -> [B, 3, 3]: F matrix
            residual -> [B, N, 1]: residual of the minimization function
        """
        device = weights.device
        weights = weights.squeeze(1).unsqueeze(2)

        ones = torch.ones_like(weights)
        if self.is_cuda:
            ones = ones.cuda()
        ## normalize the points
        pts1n, T1 = self.normalize(pts1, ones)
        pts2n, T2 = self.normalize(pts2, ones)
        # pts1n, T1 = self.normalize(pts1, weights)
        # pts2n, T2 = self.normalize(pts2, weights)

        p = torch.cat((pts2n[:,0].unsqueeze(1)*pts1n,
                       pts2n[:,1].unsqueeze(1)*pts1n,
                       pts1n), 1).permute(0,2,1)

        # # if self.normalize_SVD:
        # #     p = torch.nn.functional.normalize(p, dim=2)
        # X = p*torch.sqrt(weights)

        if self.normalize_SVD:
            p = torch.nn.functional.normalize(p, dim=2)
        # print(f"p: {p[0]}, weights: {weights[0]}")
        X = p*weights

        out_b = []
        F_vecs_list = []
        ## usually use GPU to calculate SVD and F matrix
        if self.if_cpu_svd:
            X = set_nan2zero(X)  # check if NAN
            for b in range(X.size(0)):
                # logging.info(f"X[b]: {X[b]}")
                _, _, V = torch.svd(X[b].cpu())
                F = V[:,-1].view(3,3)
                F_vecs_list.append(V[:,-1]/(V[:,-1].norm()))
                U, S, V = torch.svd(F)
                F_ = U.mm((S*self.mask.cpu()).diag()).mm(V.t())
                out_b.append(F_.unsqueeze(0))
            out = torch.cat(out_b, 0).cuda()
            F_vecs= torch.stack(F_vecs_list).cuda()
        else:
            for b in range(X.size(0)):
                _, _, V = torch.svd(X[b])
                F = V[:,-1].view(3,3)
                F_vecs_list.append(V[:,-1]/(V[:,-1].norm()))
                U, S, V = torch.svd(F)
                F_ = U.mm((S*self.mask.to(device)).diag()).mm(V.t())
                out_b.append(F_.unsqueeze(0))
            out = torch.cat(out_b, 0)
            F_vecs = torch.stack(F_vecs_list)

        if if_print:
            print(F_vecs.size(), p.size(), weights.size())
            print('----F_vecs')
            print(F_vecs[0].detach().cpu().numpy())
            print('----p')
            print(p[0].detach().cpu().numpy())
            print('----weights')
            print(weights[:2].squeeze().detach().cpu().numpy(), torch.sum(weights[:2], dim=1).squeeze().detach().cpu().numpy())

        residual = (X @ F_vecs.unsqueeze(-1)).squeeze(-1) # [B, N, 1]
        # residual_nonWeighted = (p @ F_vecs.unsqueeze(-1)).squeeze(-1) # [B, N, 1]
        # print(residual.size())
        # print(residual.norm(p=2, dim=1).size())

        out = T2.permute(0,2,1).bmm(out).bmm(T1)
        return out, residual.squeeze(-1)

    def get_unique(self, xs, topk, matches_good_unique_nums, pts1, pts2): # [B, N]
        xs_topk_list = []
        topK_indices_list = []
        pts1_list = []
        pts2_list = []

        for x, matches_good_unique_num, pt1, pt2 in zip(xs, matches_good_unique_nums, pts1, pts2):
            # x_unique = torch.unique(x) # no gradients!!!
            x_unique = x[:, :matches_good_unique_num]
            # print(x_unique_topK)
            x_unique_topK, topK_indices = torch.topk(x_unique, topk, dim=1)
            xs_topk_list.append(x_unique_topK)
            topK_indices_list.append(topK_indices.squeeze())

            pt1_topK, pt2_topK = pt1[topK_indices.squeeze(), :], pt2[topK_indices.squeeze(), :]
            pts1_list.append(pt1_topK)
            pts2_list.append(pt2_topK)
        return torch.stack(xs_topk_list), torch.stack(topK_indices_list), torch.stack(pts1_list), torch.stack(pts2_list)

    def forward(self, pts1, pts2, weights, if_print=False, matches_good_unique_num=None):
        """ check out the description in `weighted_svd`
        """
        # # print(weights.size()) # [B, 1, N]
        # # print(matches_good_unique_num)
        # weights_topK, indices_topK, pts1_topK, pts2_topK = self.get_unique(weights, 100, matches_good_unique_num, pts1, pts2)
        # # print(indices_topK, indices_topK.size())
        # # print(indices_topK.size()) # [8, 10]
        # weights_mask = torch.zeros(weights.size(0), weights.size(2), device=weights.device).float()
        # # print(indices_topK.size(), torch.max(indices_topK), weights_mask.size())
        # weights_mask = weights_mask.scatter_(1, indices_topK, 1.)
        # # print(torch.sum(weights_mask, dim=1))
        # out, residual = self.weighted_svd(pts1, pts2, weights * weights_mask.unsqueeze(1), if_print=if_print)

        # # print(pts1.size(), weights.size(), indices_topK.size()) # torch.Size([8, 1000, 3]) torch.Size([8, 1, 1000]) torch.Size([8, 100])

        out, residual = self.weighted_svd(pts1, pts2, weights, if_print=if_print)
        return out, residual

## main model
# class Norm8PointNet(nn.Module):
class DeepFNet(nn.Module):
    def __init__(self, depth, image_size, if_quality, if_img_w=False, if_goodCorresArch=False, if_tri_depth=False, if_learn_offsets=False, if_des=False, des_size=None, quality_size=0, is_cuda=True, is_test=False, if_cpu_svd=False, **params):
        super(DeepFNet, self).__init__()
        print('====Loading DeepFNet@DeepFNet.py')
        if not if_quality:
            quality_size = 0
        self.if_quality = if_quality
        if if_quality:
            print('----Quality!!!!!!@DeepFNet')
        if if_learn_offsets:
            print('----if_learn_offsets!!!!!!@DeepFNet')
        print('----CPU svd@DeepFNet!!!!!!' if if_cpu_svd else '----GPU svd@DeepFNet!!!!!!')
        self.if_des = if_des
        self.if_goodCorresArch = if_goodCorresArch
        self.if_learn_offsets = if_learn_offsets
        self.image_size = image_size # list of [H, W, 3]
        self.if_tri_depth = if_tri_depth
        self.depth_size = 1 if self.if_tri_depth else 0
        if if_tri_depth:
            print('----Tri depth!!!!!!@DeepFNet')
        if if_img_w:
            print('----Img weights!!!!!!@DeepFNet')
        self.if_img_w = if_img_w

        ## add descriptor as input
        if if_des:
            self.input_weights = ErrorEstimator(4+quality_size+des_size)
            self.update_weights = ErrorEstimator(6+quality_size+1+des_size) # +1 for the added in residual
            # self.input_weights = ErrorEstimatorFeatFusion2Head(4+quality_size, des_size)
            # self.update_weights = ErrorEstimatorFeatFusion2Head(6+quality_size+1, des_size) # +1 for the added in residual
            if if_learn_offsets:
                self.update_offsets = ErrorEstimator(6+quality_size+1+des_size, output_size=4) # +1 for the added in residual
            print('----DES feat@DeepFNet!!!!!!')
        else:
            ## deprecated (07/13/2020)
            if self.if_goodCorresArch:
                print('----goodCorresArch@DeepFNet!!!!!!')
                self.input_weights = GoodCorresNet(4+quality_size, bn=False)
                self.update_weights = GoodCorresNet(6+quality_size, bn=False)
            else:
                self.input_weights = ErrorEstimator(4+quality_size)
                self.update_weights = ErrorEstimator(4+quality_size+3+self.depth_size) # +3 for weights, epi_res and redisual, +1 for tri depth!
                if if_learn_offsets:
                    self.update_offsets = ErrorEstimator(4+quality_size+3+self.depth_size, output_size=4, if_bn=False) # +1 for the added in residual

        if is_test:
            self.input_weights.eval()
            self.update_weights.eval()
            if if_learn_offsets:
                self.update_offsets.eval()

        # self.norm = NormalizeAndExpand(is_cuda, is_test)
        # self.norm_K = NormalizeAndExpand_K(is_cuda, is_test)
        self.norm_HW = NormalizeAndExpand_HW(self.image_size, is_cuda, is_test)
        self.fit  = Fit(is_cuda, is_test, if_cpu_svd)
        self.depth = depth

        self.mask = Variable(torch.ones(3)).cuda()
        self.mask[-1] = 0

    def get_input(self, data_batch, offsets=None, iter=None):
        """ get model input from data_batch
        params:
            data_batch: batch of data
        return: 
            weight_in: [B, N, 4] matching
            pts1, pts2: matching points
            T1, T2: camera intrinsic matrix
        """
        pts = data_batch['matches_xy_ori']
        if offsets is not None:
            # print('------ ', iter)
            # print(pts.permute(0, 2, 1)[0, :2, :].clone().detach().cpu().numpy())
            # print(offsets[0, :2, :].clone().detach().cpu().numpy())
            pts = pts + offsets.permute(0, 2, 1)

        # pts1, pts2, T1, T2 = self.norm(pts) # pts: [b, N, 2] # \in [-1, 1]
        # pts1, pts2, T1, T2 = self.norm_K(pts, data_batch['K_invs']) # pts: [b, N, 2] # \in [-1, 1]
        pts1, pts2, T1, T2 = self.norm_HW(pts)
        # print(pts1.max(-1)[0].max(0)[0], pts1.min(-1)[0].min(0)[0])
        # pts1_recover = torch.inverse(T1) @ pts1
        # print(pts1_recover.max(-1)[0].max(0)[0], pts1_recover.min(-1)[0].min(0)[0])

        pts1 = pts1.permute(0,2,1)
        pts2 = pts2.permute(0,2,1)

        if self.if_quality:
            quality = data_batch['quality']
            weight_in = torch.cat(((pts1[:,:,:2]+1)/2, (pts2[:,:,:2]+1)/2, quality), 2).permute(0,2,1) # [0, 1]
        else:
            weight_in = torch.cat(((pts1[:,:,:2]+1)/2, (pts2[:,:,:2]+1)/2), 2).permute(0,2,1) # [0, 1]
        # if self.if_quality:
        #     quality = data_batch['quality']
        #     weight_in = torch.cat((pts1[:,:,:2], pts2[:,:,:2], quality), 2).permute(0,2,1) # [0, 1]
        # else:
        #     weight_in = torch.cat((pts1[:,:,:2], pts2[:,:,:2]), 2).permute(0,2,1) # [0, 1]

        # f1 = data_batch['Ks'][:, 0, 0]
        # f2 = data_batch['Ks'][:, 1, 1]
        # w2 = data_batch['Ks'][:, 0, 2]
        # h2 = data_batch['Ks'][:, 1, 2]
        # print(w2/f1)
        # print(h2/f2)
        # print(f1, f2)

        return weight_in, pts1, pts2, T1, T2

    def get_depth(self, data_batch, F_out, T1, T2):
        F_ests = T2.permute(0,2,1) @ F_out @ T1
        E_ests = data_batch['Ks'].transpose(1, 2) @ F_ests @ data_batch['Ks']
        depth_list = []
        for E_hat, K, match in zip(E_ests, data_batch['Ks'], data_batch['matches_xy_ori']):
            K = K.cpu().numpy()
            p1p2 = match.cpu().numpy()
            x1 = p1p2[:, :2]
            x2 = p1p2[:, 2:]
            num_inlier, R, t, mask_new = cv2.recoverPose(E_hat.detach().cpu().numpy().astype(np.float64), x1, x2, focal=K[0, 0], pp=(K[0, 2], K[1, 2]))
            R1 = np.eye(3)
            t1 = np.zeros((3, 1))
            M1 = np.hstack((R1, t1))
            M2 = np.hstack((R, t))
            # print(np.linalg.norm(t))
            X_tri_homo = cv2.triangulatePoints(np.matmul(K, M1), np.matmul(K, M2), x1.T, x2.T)
            X_tri = X_tri_homo[:3, :]/X_tri_homo[-1, :]
            depth = X_tri[-1, :].T
            depth_list.append(depth)
            # print(depth.flatten()[:10])
        depths = np.stack(depth_list) # [B, N]
        return torch.from_numpy(depths).unsqueeze(1).float().cuda()

    def forward(self, data_batch):
        pts_normalized_in, pts1, pts2, T1, T2 = self.get_input(data_batch)

        ## feed the matching matching for initial weights
        if self.if_des:
            # des1, des2 = data_batch['feats_im1'], data_batch['feats_im2'] # [B, D, N]
            # des_in = torch.cat((des1, des2), 1)
            # des_in = data_batch['feats_im12_var']
            des_in = data_batch['feats_im12_groupConv']
            # logits = self.input_weights(pts_normalized_in, des_in)
            logits = self.input_weights(torch.cat((pts_normalized_in, des_in), 1))
        else:
            logits = self.input_weights(pts_normalized_in)

        weights_pts = F.softmax(logits, dim=2)
        if self.if_img_w:
            weights_prod = weights_pts * data_batch['weights_im']
        else:
            weights_prod = weights_pts

        matches_good_unique_num = data_batch['matches_good_unique_nums'] # [B]
        # matches_good_unique_num = None

        # if self.if_tri_depth:
        t_scene_scale = data_batch['t_scene_scale']

        out_layers = []
        epi_res_layers = []
        residual_layers = []
        weights_layers = [weights_prod]
        logits_layers = [logits]

        # print(f"weights_prod: {weights_prod[0][0][0]}, {weights_prod[0][0].shape}")

        ## recurrent network for updated weights
        for iter in range(self.depth-1):
            ## calculate residual using current weights
            out, residual = self.fit(pts1, pts2, weights_prod, matches_good_unique_num=matches_good_unique_num)

            # if self.if_tri_depth:
            #     tri_depths = self.get_depth(data_batch, out, T1, T2) # [B, 1, N]
            #     tri_depths = torch.clamp(tri_depths * t_scene_scale, -200., 200.)

            # tri_depths = self.get_depth(data_batch, out, T1, T2) # [B, 1, N]
            # tri_depths = tri_depths * t_scene_scale
            # thres = 150.
            # tri_depths_weights = (tri_depths>0.).float() * (tri_depths<thres).float() + torch.clamp(torch.exp((thres-tri_depths)*0.05), max=1.) * (tri_depths>thres).float()

            out_layers.append(out)
            residual_layers.append(residual)
            epi_res = utils_F.compute_epi_residual(pts1, pts2, out).unsqueeze(1)
            epi_res_layers.append(epi_res)

            ## combine the input, output, and residuals for the next run
            if self.if_tri_depth:
                net_in = torch.cat((pts_normalized_in, weights_prod, epi_res, residual.unsqueeze(1), tri_depths), 1)
            else:
                # net_in = torch.cat((pts_normalized_in, weights, epi_res), 1)
                net_in = torch.cat((pts_normalized_in, weights_prod, epi_res, residual.unsqueeze(1)), 1)

            if self.if_learn_offsets:
                if self.if_des:
                    offsets = self.update_offsets(net_in, des_in)
                else:
                    offsets = self.update_offsets(net_in)
                # if iter == 0:
                offsets_accu = offsets
                # else:
                #     offsets_accu += offsets

                pts_normalized_in, pts1, pts2, T1, T2 = self.get_input(data_batch, offsets_accu, iter)

                if self.if_tri_depth:
                    net_in = torch.cat((pts_normalized_in, weights_prod, epi_res, residual.unsqueeze(1), tri_depths), 1)
                else:
                    # net_in = torch.cat((pts_normalized_in, weights, epi_res), 1)
                    net_in = torch.cat((pts_normalized_in, weights_prod, epi_res, residual.unsqueeze(1)), 1)

            if self.if_des:
                logits = self.update_weights(net_in, des_in)
            else:
                logits = self.update_weights(net_in)

            weights_pts = F.softmax(logits, dim=2)
            if self.if_img_w:
                weights_prod = weights_pts * data_batch['weights_im']
            else:
                weights_prod = weights_pts

            # weights = torch.sigmoid(logits)

            # print(tri_depths.detach().cpu().numpy())
            # print(tri_depths_weights.detach().cpu().numpy())

            # weights_prod = weights_pts * (tri_depths_weights +1e-6)

            ## add intermediate output
            weights_layers.append(weights_prod)
            logits_layers.append(logits)

        ## last run of residual
        out, residual = self.fit(pts1, pts2, weights_prod, if_print=False, matches_good_unique_num=matches_good_unique_num)
        residual_layers.append(residual)
        out_layers.append(out)

        preds = {
            # "cls_logit": cls_logit,
            "logits": logits.squeeze(1), # [batch_size, N]
            'logits_layers': logits_layers,
            'F_est': out,
            'epi_res_layers': epi_res_layers,
            'T1': T1,
            'T2': T2,
            'out_layers': out_layers,
            'pts1': pts1,
            'pts2': pts2,
            'weights': weights_prod,
            'residual_layers': residual_layers,
            'weights_layers': weights_layers, 
        }
        if self.if_learn_offsets:
            preds.update({'offsets': offsets_accu})
        if self.if_tri_depth:
            preds.update({'tri_depths': tri_depths})

        return preds


def main():
    model_params = {
        "depth": 5,
        # "img_zoom_xy": img_zoom_xy,
        "image_size": [376, 1241, 3],
        "quality_size": 0,
        "if_quality": False,
        "if_img_des_to_pointnet": False,
        "if_goodCorresArch": False,
        "if_img_feat": False,
        "if_cpu_svd": True,
        "if_learn_offsets": False,
        "if_tri_depth": False,
        "if_sample_loss": False,
    }
    net = DeepFNet(**model_params)
    print(f"deepF net: {net}")
    pass

if __name__ == "__main__":
    main()