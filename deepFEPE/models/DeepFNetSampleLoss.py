""" DeepF for sample loss
Keep but not tested (you-yi on 07/13/2020)

Authors: Rui Zhu, You-Yi Jau

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch.nn.functional import grid_sample
import numpy as np
import cv2

import dsac_tools.utils_F as utils_F # If cannot find: export KITTI_UTILS_PATH='/home/ruizhu/Documents/Projects/kitti_instance_RGBD_utils'
# import utils_F.compute_epi_residual as compute_epi_residual
# import utils_F.compute_epi_residual_non_rob as compute_epi_residual_non_rob

from models.GoodCorresNet import GoodCorresNet
# from models.ConfNet import VggDepthEstimator
# from models.ConfNet import VggDepthEstimatorOLD as VggDepthEstimator
# from models.ImageFeatNet import Conv
# from models.ImageFeatNet import VggDepthEstimatorSeperate as VggDepthEstimator
from models.ErrorEstimators import *
from batch_svd import batch_svd # https://github.com/KinglittleQ/torch-batch-svd.git


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

        H, W = image_size[0], image_size[1]
        self.T = torch.tensor([[2./W, 0., -1.], [0., 2./H, -1.], [0., 0., 1.]]).float().unsqueeze(0)

        if is_cuda:
            self.ones_b = self.ones_b.cuda()
            # self.T_b = self.T_b.cuda()
            self.T = self.T.cuda()

    def normalize(self, pts):

        ones = self.ones_b.expand(pts.size(0), pts.size(1), 1)
        pts = torch.cat((pts, ones), 2)
        pts_out = self.T @ pts.permute(0,2,1)
        return pts_out, self.T

    def forward(self, pts):
        pts1, T1 = self.normalize(pts[:,:,:2])
        pts2, T2 = self.normalize(pts[:,:,2:])

        return pts1, pts2, T1, T2

class Fit(nn.Module):
    def __init__(self, is_cuda=True, is_test=False, if_cpu_svd=False, normalize_SVD=True, if_sample_loss=False):
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
        self.if_sample_loss = if_sample_loss

        if is_cuda:
            self.ones_b = self.ones_b.cuda()
            self.zero_b = self.zero_b.cuda()
            self.T_b = self.T_b.cuda()
            self.mask = self.mask.cuda()
        self.is_cuda = is_cuda
        # self.bsvd = bsvd_torch()


    def normalize(self, pts, weights):
        device = pts.device
        T = Variable(self.T_b.to(device).expand(pts.size(0), 3, 3)).clone()
        ones = self.ones_b.to(device).expand(pts.size(0), pts.size(1), 1)

        denom = weights.sum(1)
        #

        # c = torch.mean(pts,1)
        # newpts_ = (pts - c.unsqueeze(1))
        # meandist = newpts_[:,:,:2].pow(2).sum(2).sqrt().mean(1)

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

    # def weighted_svd(self, pts1, pts2, weights):
    #     weights = weights.squeeze(1).unsqueeze(2)

    #     pts1n, T1 = self.normalize(pts1, weights)
    #     pts2n, T2 = self.normalize(pts2, weights)


    #     p = torch.cat((pts1n[:,0].unsqueeze(1)*pts2n,
    #                    pts1n[:,1].unsqueeze(1)*pts2n,
    #                    pts2n), 1).permute(0,2,1)

    #     X = p*weights

    #     out_b = []
    #     for b in range(X.size(0)):
    #         _, _, V = torch.svd(X[b])
    #         F = V[:,-1].view(3,3)
    #         U, S, V = torch.svd(F)
    #         F_ = U.mm((S*self.mask).diag()).mm(V.t())
    #         out_b.append(F_.unsqueeze(0))

    #     out = torch.cat(out_b, 0)

    #     out = T1.permute(0,2,1).bmm(out).bmm(T2)

    #     return out

    def weighted_svd(self, pts1, pts2, weights, if_print=False):
        device = weights.device
        weights = weights.squeeze(1).unsqueeze(2)

        ones = torch.ones_like(weights)
        if self.is_cuda:
            ones = ones.cuda()
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
        X = p*weights

        out_b = []
        F_vecs_list = []
        if self.if_cpu_svd:
            for b in range(X.size(0)):
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

    def weighted_svd_batch(self, pts1, pts2, weights, if_print=False):
        device = weights.device
        weights = weights.squeeze(1).unsqueeze(2)

        ones = torch.ones_like(weights)
        if self.is_cuda:
            ones = ones.cuda()
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
        X = p*weights

        Us, Ss, Vs = batch_svd(X)
        Fs = Vs[:, :, -1].view(-1, 3, 3)
        F_vecs = torch.nn.functional.normalize(Vs[:, :, -1], p=2, dim=1)
        Us, Ss, Vs = batch_svd(Fs)

        out = Us @ torch.diag_embed(Ss*self.mask.unsqueeze(0)) @ Vs.transpose(1, 2)

        # out_b = []
        # F_vecs_list = []
        # if self.if_cpu_svd:
        #     for b in range(X.size(0)):
        #         _, _, V = torch.svd(X[b].cpu())
        #         F = V[:,-1].view(3,3)
        #         F_vecs_list.append(V[:,-1]/(V[:,-1].norm()))
        #         U, S, V = torch.svd(F)
        #         F_ = U.mm((S*self.mask.cpu()).diag()).mm(V.t())
        #         out_b.append(F_.unsqueeze(0))
        #     out = torch.cat(out_b, 0).cuda()
        #     F_vecs= torch.stack(F_vecs_list).cuda()
        # else:
        #     for b in range(X.size(0)):
        #         _, _, V = torch.svd(X[b])
        #         F = V[:,-1].view(3,3)
        #         F_vecs_list.append(V[:,-1]/(V[:,-1].norm()))
        #         U, S, V = torch.svd(F)
        #         F_ = U.mm((S*self.mask.to(device)).diag()).mm(V.t())
        #         out_b.append(F_.unsqueeze(0))
        #     out = torch.cat(out_b, 0)
        #     F_vecs = torch.stack(F_vecs_list)

        # if if_print:
        #     print(F_vecs.size(), p.size(), weights.size())
        #     print('----F_vecs')
        #     print(F_vecs[0].detach().cpu().numpy())
        #     print('----p')
        #     print(p[0].detach().cpu().numpy())
        #     print('----weights')
        #     print(weights[:2].squeeze().detach().cpu().numpy(), torch.sum(weights[:2], dim=1).squeeze().detach().cpu().numpy())

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

    def forward(self, pts1, pts2, weights, if_print=False, matches_good_unique_nums=None):

        out, residual = self.weighted_svd(pts1, pts2, weights, if_print=if_print)
        out_dict = {'out': out, 'residual': residual}

        # if not(self.if_sample_loss):
        #     return out, residual, None, None

        topK = 20
        selects_each_sample = 100
        # print(weights.size()) # [B, 1, N]
        weights_topK, indices_topK, pts1_topK, pts2_topK = self.get_unique(weights, topK, matches_good_unique_nums, pts1, pts2)
        # print(indices_topK, indices_topK.size())
        # print(indices_topK.size()) # [8, 10]
        weights_mask = torch.zeros(weights.size(0), weights.size(2), device=weights.device).float() # [B, topK]
        # print(indices_topK.size(), torch.max(indices_topK), weights_mask.size())
        weights_mask = weights_mask.scatter_(1, indices_topK, 1.)
        # print(torch.sum(weights_mask, dim=1))

        # print(pts1.size(), weights.size(), indices_topK.size()) # torch.Size([8, 1000, 3]) torch.Size([8, 1, 1000]) torch.Size([8, 100])

        pts1_topK = torch.gather(pts1, 1, indices_topK.unsqueeze(-1).expand(-1, -1, 3))
        pts2_topK = torch.gather(pts2, 1, indices_topK.unsqueeze(-1).expand(-1, -1, 3))
        weights_topK = torch.gather(weights, 2, indices_topK.unsqueeze(1))
        # a = torch.index_select(pts1, 1, indices_topK.unsqueeze(-1))
        # mask_select = weights_mask.byte().unsqueeze(-1)
        # a = torch.masked_select(pts1, mask_select)

        # out_topK, residual_topK = self.weighted_svd(pts1_topK, pts2_topK, weights_topK, if_print=if_print)
        out_topK, residual_topK = self.weighted_svd_batch(pts1_topK, pts2_topK, weights_topK, if_print=if_print)
        out_dict.update({'out_topK': out_topK, 'residual_topK': residual_topK})

        # out, residual = self.weighted_svd(pts1, pts2, weights * weights_mask.unsqueeze(1), if_print=if_print)
        

        out_sample_selected_list = []
        weights_sample_selected_accu_list = []
        for batch_idx, (matches_good_unique_num, weights_sample) in enumerate(zip(matches_good_unique_nums.cpu().numpy(), weights.detach().cpu().numpy())):
            selected_corres_idx_per_sample_list = []
            p = weights_sample.flatten()[:matches_good_unique_num]
            p = p / np.sum(p)
            for select_idx in range(selects_each_sample):
                selected_corres_idx = np.random.choice(matches_good_unique_num, topK, p=p)
                # selected_corres_idx = np.random.choice(matches_good_unique_num, topK)
                selected_corres_idx_per_sample_list.append(selected_corres_idx)
            selected_corres_idx_per_sample = np.stack(selected_corres_idx_per_sample_list) # [selects_each_sample, topK]
            pts1_sample = pts1[batch_idx:batch_idx+1].expand(selects_each_sample, -1, -1)
            pts1_sample_selected = torch.gather(pts1_sample, 1, torch.from_numpy(selected_corres_idx_per_sample).unsqueeze(-1).expand(-1, -1, 3).cuda()) # [selects_each_sample, topK, 3]
            pts2_sample = pts2[batch_idx:batch_idx+1].expand(selects_each_sample, -1, -1)
            pts2_sample_selected = torch.gather(pts2_sample, 1, torch.from_numpy(selected_corres_idx_per_sample).unsqueeze(-1).expand(-1, -1, 3).cuda()) # [selects_each_sample, topK, 3]
            weights_sample = weights[batch_idx:batch_idx+1].expand(selects_each_sample, -1, -1)
            weights_sample_selected = torch.gather(weights_sample, 2, torch.from_numpy(selected_corres_idx_per_sample).unsqueeze(1).cuda()) # [selects_each_sample, 1, topK]
            weights_sample_selected_normalized = torch.nn.functional.normalize(weights_sample_selected, p=1, dim=2) # [selects_each_sample, 1, topK]

            weights_sample_selected_accu = torch.prod(weights_sample_selected * 1000., dim=2) # [selects_each_sample, 1]
            weights_sample_selected_accu = weights_sample_selected_accu / (torch.sum(weights_sample_selected_accu)+1e-10)
            # print(weights_sample_selected_accu, torch.sum(weights_sample_selected_accu))
            weights_sample_selected_accu_list.append(weights_sample_selected_accu)

            # out_sample_selected, _ = self.weighted_svd(pts1_sample_selected, pts2_sample_selected, weights_sample_selected_normalized, if_print=False) # [selects_each_sample, 3, 3]
            out_sample_selected, _ = self.weighted_svd_batch(pts1_sample_selected, pts2_sample_selected, weights_sample_selected, if_print=False) # [selects_each_sample, 3, 3]

            out_sample_selected_list.append(out_sample_selected)
        
        out_sample_selected_batch = torch.stack(out_sample_selected_list) # [B, selects_each_sample, 3, 3]
        weights_sample_selected_accu_batch = torch.stack(weights_sample_selected_accu_list) # [B, selects_each_sample, 1]

        # return out_topK, residual_topK, out_sample_selected_batch, weights_sample_selected_accu_batch

        out_dict.update({'out_sample_selected_batch': out_sample_selected_batch, 'weights_sample_selected_accu_batch': weights_sample_selected_accu_batch})


        return out_dict

class Norm8PointNet(nn.Module):
    def __init__(self, depth, image_size, if_quality, if_goodCorresArch=False, if_tri_depth=False, if_sample_loss=False, if_learn_offsets=False, if_des=False, des_size=None, quality_size=0, is_cuda=True, is_test=False, if_cpu_svd=False, **params):
        super(Norm8PointNet, self).__init__()
        print('====Loading Norm8PointNet@DeepFNetSampleLoss.py')
        if not if_quality:
            quality_size = 0
        self.if_quality = if_quality
        if if_quality:
            print('----Quality!!!!!!@Norm8PointNet')
        if if_learn_offsets:
            print('----if_learn_offsets!!!!!!@Norm8PointNet')
        print('----CPU svd@Norm8PointNet!!!!!!' if if_cpu_svd else '----GPU svd@Norm8PointNet!!!!!!')
        self.if_des = if_des
        self.if_goodCorresArch = if_goodCorresArch
        self.if_learn_offsets = if_learn_offsets
        self.image_size = image_size # list of [H, W, 3]
        self.if_tri_depth = if_tri_depth
        self.depth_size = 1 if self.if_tri_depth else 0
        if if_tri_depth:
            print('----Tri depth!!!!!!@Norm8PointNet')
        self.if_sample_loss = if_sample_loss
        if if_sample_loss:
            print('----if_sample_loss!!!!!!@Norm8PointNet')



        if if_des:
            # self.input_weights = ErrorEstimatorDes(4+quality_size, des_size)
            # self.update_weights = ErrorEstimatorDes(6+quality_size, des_size)

            # self.input_weights = ErrorEstimatorFeatFusion(4+quality_size, des_size)
            # self.update_weights = ErrorEstimatorFeatFusion(6+quality_size+1, des_size) # +1 for the added in residual
            # if if_learn_offsets:
            #     self.update_offsets = ErrorEstimatorFeatFusion(6+quality_size+1, des_size, output_size=4) # +1 for the added in residual

            self.input_weights = ErrorEstimator(4+quality_size+des_size)
            self.update_weights = ErrorEstimator(6+quality_size+1+des_size) # +1 for the added in residual
            # self.input_weights = ErrorEstimatorFeatFusion2Head(4+quality_size, des_size)
            # self.update_weights = ErrorEstimatorFeatFusion2Head(6+quality_size+1, des_size) # +1 for the added in residual
            if if_learn_offsets:
                self.update_offsets = ErrorEstimator(6+quality_size+1+des_size, output_size=4) # +1 for the added in residual
            print('----DES feat@Norm8PointNet!!!!!!')
        else:
            if self.if_goodCorresArch:
                print('----goodCorresArch@Norm8PointNet!!!!!!')
                self.input_weights = GoodCorresNet(4+quality_size, bn=False)
                self.update_weights = GoodCorresNet(6+quality_size, bn=False)
            else:
                self.input_weights = ErrorEstimator(4+quality_size)
                self.update_weights = ErrorEstimator(4+quality_size+3+self.depth_size) # +3 for weights, epi_res and redisual, +1 for tri depth!
                if if_learn_offsets:
                    self.update_offsets = ErrorEstimator(4+quality_size+2+self.depth_size, output_size=4, if_bn=False) # +1 for the added in residual


        if is_test:
            self.input_weights.eval()
            self.update_weights.eval()
            if if_learn_offsets:
                self.update_offsets.eval()

        self.norm = NormalizeAndExpand(is_cuda, is_test)
        self.norm_K = NormalizeAndExpand_K(is_cuda, is_test)
        self.norm_HW = NormalizeAndExpand_HW(self.image_size, is_cuda, is_test)
        self.fit  = Fit(is_cuda, is_test, if_cpu_svd, if_sample_loss=if_sample_loss)
        self.depth = depth

        self.mask = Variable(torch.ones(3)).cuda()
        self.mask[-1] = 0

    def get_input(self, data_batch, offsets=None, iter=None):
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

        if self.if_des:
            # des1, des2 = data_batch['feats_im1'], data_batch['feats_im2'] # [B, D, N]
            # des_in = torch.cat((des1, des2), 1)
            # des_in = data_batch['feats_im12_var']
            des_in = data_batch['feats_im12_groupConv']
            # logits = self.input_weights(pts_normalized_in, des_in)
            logits = self.input_weights(torch.cat((pts_normalized_in, des_in), 1))
        else:
            logits = self.input_weights(pts_normalized_in)

        weights = F.softmax(logits, dim=2)
        # weights = torch.sigmoid(logits)

        matches_good_unique_nums = data_batch['matches_good_unique_nums'] # [B]
        # matches_good_unique_num = None

        if self.if_tri_depth:
            t_scene_scale = data_batch['t_scene_scale']

        out_layers = []
        out_topK_layers = []
        epi_res_layers = []
        residual_layers = []
        weights_layers = [weights]
        logits_layers = [logits]
        out_sample_selected_batch_layers = []
        weights_sample_selected_accu_batch_layers = []


        for iter in range(self.depth-1):

            out_dict = self.fit(pts1, pts2, weights, matches_good_unique_nums=matches_good_unique_nums)
            out, residual = out_dict['out'], out_dict['residual']
            residual_layers.append(residual)
            out_layers.append(out)
            out_topK_layers.append(out_dict['out_topK'])
            out_sample_selected_batch_layers.append(out_dict['out_sample_selected_batch'])
            weights_sample_selected_accu_batch_layers.append(out_dict['weights_sample_selected_accu_batch'])

            if self.if_tri_depth:
                tri_depths = self.get_depth(data_batch, out, T1, T2) # [B, 1, N]
                tri_depths = torch.clamp(tri_depths * t_scene_scale, -150., 150.)
            epi_res = utils_F.compute_epi_residual(pts1, pts2, out).unsqueeze(1)
            epi_res_layers.append(epi_res)

            if self.if_tri_depth:
                net_in = torch.cat((pts_normalized_in, weights, epi_res, tri_depths), 1)
            else:
                # net_in = torch.cat((pts_normalized_in, weights, epi_res), 1)
                net_in = torch.cat((pts_normalized_in, weights, epi_res), 1)

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
                    net_in = torch.cat((pts_normalized_in, weights, epi_res, tri_depths), 1)
                else:
                    # net_in = torch.cat((pts_normalized_in, weights, epi_res), 1)
                    net_in = torch.cat((pts_normalized_in, weights, epi_res), 1)

            if self.if_des:
                logits = self.update_weights(net_in, des_in)
            else:
                logits = self.update_weights(net_in)

            weights = F.softmax(logits, dim=2)
            # weights = torch.sigmoid(logits)
            weights_layers.append(weights)
            logits_layers.append(logits)

        out_dict = self.fit(pts1, pts2, weights, matches_good_unique_nums=matches_good_unique_nums)
        out, residual = out_dict['out'], out_dict['residual']
        residual_layers.append(residual)
        out_layers.append(out)
        out_topK_layers.append(out_dict['out_topK'])
        out_sample_selected_batch_layers.append(out_dict['out_sample_selected_batch'])
        weights_sample_selected_accu_batch_layers.append(out_dict['weights_sample_selected_accu_batch'])

        preds = {
            # "cls_logit": cls_logit,
            "logits": logits.squeeze(1), # [batch_size, N]
            'logits_layers': logits_layers,
            'F_est': out,
            'epi_res_layers': epi_res_layers,
            'T1': T1,
            'T2': T2,
            'out_layers': out_layers,
            'out_topK_layers': out_topK_layers,
            'pts1': pts1,
            'pts2': pts2,
            'weights': weights,
            'residual_layers': residual_layers,
            'weights_layers': weights_layers, 
            'out_sample_selected_batch_layers': out_sample_selected_batch_layers, 
            'weights_sample_selected_accu_batch_layers': weights_sample_selected_accu_batch_layers
        }
        if self.if_learn_offsets:
            preds.update({'offsets': offsets_accu})
        if self.if_tri_depth:
            preds.update({'tri_depths': tri_depths})

        return preds


# class Norm8PointNet_bkg(nn.Module):
#     def __init__(self, depth, if_quality, if_goodCorresArch=False, if_learn_offsets=False, if_des=False, des_size=None, quality_size=0, is_cuda=True, is_test=False, if_cpu_svd=False, **params):
#         super(Norm8PointNet, self).__init__()
#         print('====Loading Norm8PointNet@DeepFNet.py')
#         if not if_quality:
#             quality_size = 0
#         self.if_quality = if_quality
#         if if_quality:
#             print('----Quality!!!!!!')
#         self.if_des = if_des
#         self.if_goodCorresArch = if_goodCorresArch

#         if if_des:
#             # self.input_weights = ErrorEstimatorDes(4+quality_size, des_size)
#             # self.update_weights = ErrorEstimatorDes(6+quality_size, des_size)
#             self.input_weights = ErrorEstimatorFeatFusion(4+quality_size, des_size*2)
#             self.update_weights = ErrorEstimatorFeatFusion(6+quality_size, des_size*2)
#             if if_learn_offsets:
#                 self.update_offsets = ErrorEstimatorFeatFusion(6+quality_size+1, des_size*2, output_size=4, if_bn=False)
#             print('----DES feat@Norm8PointNet!!!!!!')
#         else:
#             if self.if_goodCorresArch:
#                 print('----goodCorresArch@Norm8PointNet!!!!!!')
#                 self.input_weights = GoodCorresNet(4+quality_size)
#                 self.update_weights = GoodCorresNet(6+quality_size)
#             else:
#                 self.input_weights = ErrorEstimator(4+quality_size)
#                 self.update_weights = ErrorEstimator(6+quality_size)
#                 if if_learn_offsets:
#                     self.update_offsets = ErrorEstimator(6+quality_size+1, output_size=4, if_bn=False)


#         if is_test:
#             self.input_weights.eval()
#             self.update_weights.eval()
#             if if_learn_offsets:
#                 self.update_offsets.eval()

#         self.norm = NormalizeAndExpand(is_cuda, is_test)
#         self.norm_K = NormalizeAndExpand_K(is_cuda, is_test)
#         self.fit  = Fit(is_cuda, is_test, if_cpu_svd)
#         print('----CPU svd!!!!!!' if if_cpu_svd else '----GPU svd!!!!!!')
#         self.depth = depth

#         self.mask = Variable(torch.ones(3)).cuda()
#         self.mask[-1] = 0


#     def forward(self, data_batch):
#         pts = data_batch['matches_xy_ori']
#         # pts1, pts2, T1, T2 = self.norm(pts) # pts: [b, N, 2] # \in [-1, 1]
#         pts1, pts2, T1, T2 = self.norm_K(pts, data_batch['K_invs']) # pts: [b, N, 2] # \in [-1, 1]
#         if self.if_des:
#             # des1, des2 = data_batch['des1'].transpose(1, 2), data_batch['des2'].transpose(1, 2)
#             des1, des2 = data_batch['feats_im1'], data_batch['feats_im2'] # [B, D, N]
#             des_in = torch.cat((des1, des2), 1)

#         pts1 = pts1.permute(0,2,1)
#         pts2 = pts2.permute(0,2,1)

#         if self.if_quality:
#             quality = data_batch['quality']
#             weight_in = torch.cat(((pts1[:,:,:2]+1)/2, (pts2[:,:,:2]+1)/2, quality), 2).permute(0,2,1) # [0, 1]
#         else:
#             weight_in = torch.cat(((pts1[:,:,:2]+1)/2, (pts2[:,:,:2]+1)/2), 2).permute(0,2,1) # [0, 1]

#         if self.if_des:
#             logits = self.input_weights(weight_in, des_in)
#         else:
#             logits = self.input_weights(weight_in)

#         weights = F.softmax(logits, dim=2)
#         # weights = torch.sigmoid(logits)

#         out_layers = []
#         epi_res_layers = []
#         residual_layers = []
#         weights_layers = [weights]


#         for iter in range(self.depth-1):
#             out, residual = self.fit(pts1, pts2, weights)
#             out_layers.append(out)
#             residual_layers.append(residual)
#             epi_res = utils_F.compute_epi_residual(pts1, pts2, out).unsqueeze(1)
#             epi_res_layers.append(epi_res)

#             net_in = torch.cat((weight_in, weights, epi_res), 1)

#             if self.if_des:
#                 logits = self.update_weights(net_in, des_in)
#             else:
#                 logits = self.update_weights(net_in)

#             weights = F.softmax(logits, dim=2)
#             # weights = torch.sigmoid(logits)
#             weights_layers.append(weights)

#         out, residual = self.fit(pts1, pts2, weights, if_print=False)
#         residual_layers.append(residual)

#         preds = {
#             # "cls_logit": cls_logit,
#             "logits": logits.squeeze(1), # [batch_size, N]
#             'F_est': out,
#             'epi_res_layers': epi_res_layers,
#             'T1': T1,
#             'T2': T2,
#             'out_layers': out_layers,
#             'pts1': pts1,
#             'pts2': pts2,
#             'weights': weights,
#             'residual_layers': residual_layers,
#             'weights_layers': weights_layers
#         }

#         return preds


# class Norm8PointNetMixWeights(nn.Module):
#     def __init__(self, depth, if_quality, if_des, if_goodCorresArch, quality_size=0, is_cuda=True, is_test=False):
#         super(Norm8PointNetMixWeights, self).__init__()
#         if not if_quality:
#             quality_size = 0
#         self.if_quality = if_quality
#         if if_quality:
#             print('----Quality!!!!!!')
#         self.if_des = if_des
#         self.if_goodCorresArch = if_goodCorresArch

#         self.input_weights = ErrorEstimator(4+quality_size)
#         self.update_weights = ErrorEstimator(6+quality_size)

#         if is_test:
#             self.input_weights.eval()
#             self.update_weights.eval()

#         self.norm = NormalizeAndExpand(is_cuda, is_test)
#         self.norm_K = NormalizeAndExpand_K(is_cuda, is_test)
#         self.fit  = Fit(is_cuda, is_test)
#         self.depth = depth

#         self.mask = Variable(torch.ones(3)).cuda()
#         self.mask[-1] = 0


#     def forward(self, data_batch):
#         pts = data_batch['matches_xy_ori']
#         # pts1, pts2, T1, T2 = self.norm(pts) # pts: [b, N, 2] # \in [-1, 1]
#         pts1, pts2, T1, T2 = self.norm_K(pts, data_batch['K_invs']) # pts: [b, N, 2] # \in [-1, 1]
#         pts1 = pts1.permute(0,2,1)
#         pts2 = pts2.permute(0,2,1)

#         weights_im1 = data_batch['feats_im1'] # [B, 1, N]
#         weights_im2 = data_batch['feats_im2'] # [B, 1, N]

#         if self.if_quality:
#             quality = data_batch['quality']
#             weight_in = torch.cat(((pts1[:,:,:2]+1)/2, (pts2[:,:,:2]+1)/2, quality), 2).permute(0,2,1) # [B, D, N]
#         else:
#             weight_in = torch.cat(((pts1[:,:,:2]+1)/2, (pts2[:,:,:2]+1)/2), 2).permute(0,2,1) # [0, 1]

#         logits = self.input_weights(weight_in)

#         weights = F.softmax(logits, dim=2) # [B, 1, N]
#         weights = weights * weights_im1 * weights_im2


#         out_a = []

#         for iter in range(self.depth-1):
#             out = self.fit(pts1, pts2, weights)
#             out_a.append(out)
#             res = utils_F.compute_epi_residual(pts1, pts2, out, clamp_at=0.05).unsqueeze(1)

#             # res_np = res.detach().cpu().numpy().squeeze()
#             # print(res_np, res_np.shape, np.amax(res_np, 1), np.amin(res_np, 1), np.mean(res_np, 1), np.median(res_np, 1))

#             net_in = torch.cat((weight_in, weights, res), 1)

#             logits = self.update_weights(net_in)

#             weights = F.softmax(logits, dim=2) * weights_im1 * weights_im2

#         out = self.fit(pts1, pts2, weights)

#         preds = {
#             # "cls_logit": cls_logit,
#             "logits": logits.squeeze(1), # [batch_size, N]
#             'F_est': out,
#             'res': weights.squeeze(1),
#             'T1': T1,
#             'T2': T2,
#             'out_a': out_a,
#             'pts1': pts1,
#             'pts2': pts2,
#             'weights': weights
#         }

#         return preds




# class NWeightMixer(nn.Module):
#     def __init__(self, input_size):
#         super(NWeightMixer, self).__init__()

#         inplace = True
#         hasbias = True
#         learn_affine = True
#         self.fw = nn.Sequential(
#             nn.Conv1d(input_size, 16, kernel_size=1, bias=hasbias),
#             # nn.InstanceNorm1d(64, affine=learn_affine),
#             nn.LeakyReLU(inplace=inplace),
#             nn.Conv1d(16,32, kernel_size=1, bias=hasbias),
#             # nn.InstanceNorm1d(128, affine=learn_affine),
#             nn.LeakyReLU(inplace=inplace),
#             nn.Conv1d(32,16,kernel_size=1, bias=hasbias),
#             # nn.InstanceNorm1d(1024, affine=learn_affine),
#             nn.LeakyReLU(inplace=inplace),
#             nn.Conv1d(16,1, kernel_size=1, bias=hasbias),
#             nn.

#     def forward(self, data):
#         # print('ErrorEstimator')
#         return self.fw(data)
