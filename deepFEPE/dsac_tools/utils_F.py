import torch
import numpy as np
import cv2
import sys, os
# from numpy import *
import scipy
import random
from six.moves import xrange
import operator
import deepFEPE.dsac_tools.utils_misc as utils_misc
import deepFEPE.dsac_tools.utils_geo as utils_geo
import deepFEPE.dsac_tools.utils_vis as utils_vis
# from batch_svd import batch_svd # https://github.com/KinglittleQ/torch-batch-svd.git

def _normalize_XY(X, Y):
    """ The Hartley normalization. Following https://github.com/marktao99/python/blob/da2682f8832483650b85b0be295ae7eaf179fcc5/CVP/samples/sfm.py#L157 
    corrected with https://www.mathworks.com/matlabcentral/fileexchange/27541-fundamental-matrix-computation
    and https://en.wikipedia.org/wiki/Eight-point_algorithm#The_normalized_eight-point_algorithm """
    if X.size()[0] != Y.size()[0]:
        raise ValueError("Number of points don't match.")
    X = utils_misc._homo(X)
    mean_1 = torch.mean(X[:, :2], dim=0, keepdim=True)
    S1 = np.sqrt(2) / torch.mean(torch.norm(X[:, :2]-mean_1, 2, dim=1))
    # print(mean_1.size(), S1.size())
    T1 = torch.tensor([[S1,0,-S1*mean_1[0, 0]],[0,S1,-S1*mean_1[0, 1]],[0,0,1]], device=X.device)
    X_normalized = utils_misc._de_homo(torch.mm(T1, X.t()).t()) # ideally zero mean (x, y), and sqrt(2) average norm

    # xxx = X_normalized.numpy()
    # print(np.mean(xxx, axis=0))
    # print(np.mean(np.linalg.norm(xxx, 2, axis=1)))

    Y = utils_misc._homo(Y)
    mean_2 = torch.mean(Y[:, :2], dim=0, keepdim=True)
    S2 = np.sqrt(2) / torch.mean(torch.norm(Y[:, :2]-mean_2, 2, dim=1))
    T2 = torch.tensor([[S2,0,-S2*mean_2[0, 0]],[0,S2,-S2*mean_2[0, 1]],[0,0,1]], device=X.device)
    Y_normalized = utils_misc._de_homo(torch.mm(T2, Y.t()).t())

    return X_normalized, Y_normalized, T1, T2

def _normalize_XY_batch(X, Y):
    """ The Hartley normalization. Following https://github.com/marktao99/python/blob/da2682f8832483650b85b0be295ae7eaf179fcc5/CVP/samples/sfm.py#L157 
    corrected with https://www.mathworks.com/matlabcentral/fileexchange/27541-fundamental-matrix-computation
    and https://en.wikipedia.org/wiki/Eight-point_algorithm#The_normalized_eight-point_algorithm """
    # X: [batch_size, N, 2]
    if X.size()[1] != Y.size()[1]:
        raise ValueError("Number of points don't match.")
    X = utils_misc._homo(X)
    mean_1s = torch.mean(X[:, :, :2], dim=1, keepdim=True)
    S1s = np.sqrt(2) / torch.mean(torch.norm(X[:, :, :2]-mean_1s, 2, dim=2), dim=1)
    T1_list = []
    for S1, mean_1 in zip(S1s, mean_1s):
        T1_list.append(torch.tensor([[S1,0,-S1*mean_1[0, 0]],[0,S1,-S1*mean_1[0, 1]],[0,0,1]], device=X.device))
    T1s = torch.stack(T1_list)
    X_normalized = utils_misc._de_homo(torch.bmm(T1s, X.transpose(1, 2)).transpose(1, 2)) # ideally zero mean (x, y), and sqrt(2) average norm

    # xxx = X_normalized.numpy()
    # print(np.mean(xxx, axis=0))
    # print(np.mean(np.linalg.norm(xxx, 2, axis=1)))

    Y = utils_misc._homo(Y)
    mean_2s = torch.mean(Y[:, :, :2], dim=1, keepdim=True)
    S2s = np.sqrt(2) / torch.mean(torch.norm(Y[:, :,  :2]-mean_2s, 2, dim=2), dim=1)
    T2_list = []
    for S2, mean_2 in zip(S2s, mean_2s):
        T2_list.append(torch.tensor([[S2,0,-S2*mean_2[0, 0]],[0,S2,-S2*mean_2[0, 1]],[0,0,1]], device=X.device))
    T2s = torch.stack(T2_list)
    Y_normalized = utils_misc._de_homo(torch.bmm(T2s, Y.transpose(1, 2)).transpose(1, 2))

    return X_normalized, Y_normalized, T1s, T2s

# def E_from_XY(X, Y):
#     # X, Y: [N, 2]
#     xx = torch.cat([X.t(), Y.t()], dim=0)
#     # print(xx.size())
#     X = torch.stack([
#         xx[2, :] * xx[0, :], xx[2, :] * xx[1, :], xx[2, :],
#         xx[3, :] * xx[0, :], xx[3, :] * xx[1, :], xx[3, :],
#         xx[0, :], xx[1, :], torch.ones_like(xx[0, :])
#     ], dim=0).t()
#     XwX = torch.matmul(X.t(), X)
#     # print("XwX shape = {}".format(XwX.shape))

#     # Recover essential matrix from self-adjoing eigen
#     e, v = torch.eig(XwX, eigenvectors=True)
#     # print(t)
#     # print('----E_gt', E.numpy())
#     E_recover = v[:, 8].reshape((3, 3))
#     print(E_recover.numpy())
#     # E_recover_rescale = E_recover / torch.norm(E_recover) * torch.norm(E)
#     # print('-E_recover', E_recover_rescale.numpy())
#     U, D, V = torch.svd(E_recover)
#     diag_sing = torch.diag(torch.tensor([1., 1., 0.], dtype=torch.float64))
#     E_recover_hat = torch.mm(U, torch.mm(diag_sing, V.t()))
#     # E_recover_hat_rescale = E_recover_hat / torch.norm(E_recover_hat) * torch.norm(E)
#     # print('--E_recover_hat', E_recover_hat_rescale.numpy())

#     return E_recover_hat

# def _E_from_XY(X, Y, K):
#     F = _F_from_XY(X, Y)
#     E = _F_to_E(F, K)
#     return E

def _E_from_XY(X, Y, K, W=None, if_normzliedK=False, normalize=True, show_debug=False): # Ref: https://github.com/marktao99/python/blob/master/CVP/samples/sfm.py#L55
    """ Normalized Eight Point Algorithom for E: [Manmohan] In practice, one would transform the data points by K^{-1}, then do a Hartley normalization, then estimate the F matrix (which is now E matrix), then set the singular value conditions, then denormalize. Note that it's better to set singular values first, then denormalize.
        X, Y: [N, 2] """
    if if_normzliedK:
        X_normalizedK = X
        Y_normalizedK = Y
    else:
        X_normalizedK = utils_misc._de_homo(torch.mm(torch.inverse(K), utils_misc._homo(X).t()).t())
        Y_normalizedK = utils_misc._de_homo(torch.mm(torch.inverse(K), utils_misc._homo(Y).t()).t())

    if normalize:
        X, Y, T1, T2 = _normalize_XY(X_normalizedK, Y_normalizedK)
    else:
        X, Y = X_normalizedK, Y_normalizedK
    # print(T1)
    # print(T2)
    # print(X)

    xx = torch.cat([X.t(), Y.t()], dim=0)
    XX = torch.stack([
        xx[2, :] * xx[0, :], xx[2, :] * xx[1, :], xx[2, :],
        xx[3, :] * xx[0, :], xx[3, :] * xx[1, :], xx[3, :],
        xx[0, :], xx[1, :], torch.ones_like(xx[0, :])
    ], dim=0).t() # [N, 9]
    # print(XX.size())
    if W is not None:
        XX = torch.mm(W, XX) # [N, 9]
    # print(XX[:2])
    U, D, V = torch.svd(XX, some=True)
    if show_debug:
        print('[info.Debug @_E_from_XY] Singualr values of XX:\n', D.numpy())

    # U_np, D_np, V_np = np.linalg.svd(XX.numpy())

    F_recover = torch.reshape(V[:, -1], (3, 3))
    # print('-', F_recover)

    FU, FD, FV= torch.svd(F_recover, some=True)
    if show_debug:
        print('[info.Debug @_E_from_XY] Singular values for recovered E(F):\n', FD.numpy())

    # FDnew = torch.diag(FD);
    # FDnew[2, 2] = 0;
    # F_recover_sing = torch.mm(FU, torch.mm(FDnew, FV.t()))
    S_110 = torch.diag(torch.tensor([1., 1., 0.], dtype=FU.dtype, device=FU.device))
    E_recover_110 = torch.mm(FU, torch.mm(S_110, FV.t()))
    # F_recover_sing_rescale = F_recover_sing / torch.norm(F_recover_sing) * torch.norm(F)

    # print(E_recover_110)
    if normalize:
        E_recover_110 = torch.mm(T2.t(), torch.mm(E_recover_110, T1))
    return E_recover_110

def _E_from_XY_batch(X, Y, K, W=None, if_normzliedK=False, normalize=True, show_debug=False): # Ref: https://github.com/marktao99/python/blob/master/CVP/samples/sfm.py#L55
    from batch_svd import batch_svd # https://github.com/KinglittleQ/torch-batch-svd.git
    """ Normalized Eight Point Algorithom for E: [Manmohan] In practice, one would transform the data points by K^{-1}, then do a Hartley normalization, then estimate the F matrix (which is now E matrix), then set the singular value conditions, then denormalize. Note that it's better to set singular values first, then denormalize.
        X, Y: [N, 2] """
    assert X.dtype==torch.float32, 'batch_svd currently only supports torch.float32!'
    if if_normzliedK:
        X_normalizedK = X.float()
        Y_normalizedK = Y.float()
    else:
        X_normalizedK = utils_misc._de_homo(torch.bmm(torch.inverse(K), utils_misc._homo(X).transpose(1, 2)).transpose(1, 2)).float()
        Y_normalizedK = utils_misc._de_homo(torch.bmm(torch.inverse(K), utils_misc._homo(Y).transpose(1, 2)).transpose(1, 2)).float()

    # assert normalize==False, 'Not supported in batch mode yet!'
    if normalize:
        X, Y, T1, T2 = _normalize_XY_batch(X_normalizedK, Y_normalizedK)
    else:
        X, Y = X_normalizedK, Y_normalizedK

    # print(T1)
    # print(T2)
    # print(X)

    xx = torch.cat([X, Y], dim=2)
    XX = torch.stack([
        xx[:, :, 2] * xx[:, :, 0], xx[:, :, 2] * xx[:, :, 1], xx[:, :, 2],
        xx[:, :, 3] * xx[:, :, 0], xx[:, :, 3] * xx[:, :, 1], xx[:, :, 3],
        xx[:, :, 0], xx[:, :, 1], torch.ones_like(xx[:, :, 0])
    ], dim=2)

    if W is not None:
        XX = torch.bmm(W, XX)
    # U, D, V = torch.svd(XX, some=False)
    # print(XX[0, :2])
    # print(XX.size())
    # U, D, V = batch_svd(XX)
    V_list = []
    for XX_single in XX:
        _, _, V_single = torch.svd(XX_single, some=True)
        V_list.append(V_single[:, -1])
    V_last_col = torch.stack(V_list)
    # print(V_last_col.size(), '----')

    if show_debug:
        print('[info.Debug @_E_from_XY] Singualr values of XX:\n', D[0].numpy())

    # F_recover = torch.reshape(V[:, :, -1], (-1, 3, 3))
    F_recover = V_last_col.view(-1, 3, 3)

    # FU, FD, FV= torch.svd(F_recover, some=False)
    FU, FD, FV= batch_svd(F_recover)

    if show_debug:
        print('[info.Debug @_E_from_XY] Singular values for recovered E(F):\n', FD[0].numpy())

    # FDnew = torch.diag(FD);
    # FDnew[2, 2] = 0;
    # F_recover_sing = torch.mm(FU, torch.mm(FDnew, FV.t()))
    S_110 = torch.diag(torch.tensor([1., 1., 0.], dtype=FU.dtype, device=FU.device)).unsqueeze(0).expand(FV.size()[0], -1, -1)

    E_recover_110 = torch.bmm(FU, torch.bmm(S_110, FV.transpose(1, 2)))
    # F_recover_sing_rescale = F_recover_sing / torch.norm(F_recover_sing) * torch.norm(F)
    # print(E_recover_110)
    if normalize:
        E_recover_110 = torch.bmm(T2.transpose(1, 2), torch.bmm(E_recover_110, T1))
    return -E_recover_110

def _F_from_XY(X, Y, W=None, normalize=True, show_debug=False): # Ref: https://github.com/marktao99/python/blob/master/CVP/samples/sfm.py#L55
    # X, Y: [N, 2]
    if normalize:
        # print(X.t().numpy())
        X, Y, T1, T2 = _normalize_XY(X, Y)

    xx = torch.cat([X.t(), Y.t()], dim=0)
    # print(xx.size())
    # print(xx.size())
    XX = torch.stack([
        xx[2, :] * xx[0, :], xx[2, :] * xx[1, :], xx[2, :],
        xx[3, :] * xx[0, :], xx[3, :] * xx[1, :], xx[3, :],
        xx[0, :], xx[1, :], torch.ones_like(xx[0, :])
    ], dim=0).t()
    if W is not None:
        XX = torch.mm(W, XX)
    U, D, V = torch.svd(XX, some=True)
    # print(XX.size(), U.size(), D.size(), V.size())
    # print(V.size(), V[:, -1].reshape([3, 3]).numpy())
    # print(D.numpy())

    # u, d, v = batch_svd(XX.cuda().float().unsqueeze(0))
    # print(v.size(), v[0, :, -1].reshape([3, 3]).cpu().numpy())
    # print(d.cpu().numpy().flatten())

    if show_debug:
        print('[info.Debug@_F_from_XY] Singualr values of XX:\n', D.numpy())
    # print(D.numpy())
    # print(V.numpy().T, V.numpy().shape)
    # print(U.size(), D.size(), V.size(), X.size())
    # print(V[:, -1].numpy()))

    # U_np, D_np, V_np = np.linalg.svd(XX.numpy())
    # V_np = torch.from_numpy(V_np)
    # print(V[-1].numpy())

    F_recover = torch.reshape(V[:, -1], (3, 3))
    # F_recover = torch.reshape(v[0, :, -1].cpu(), (3, 3))

    # return F_recover, np.reshape(V_np[-1], (3, 3))

    # F_recover_rescale = F_recover / torch.norm(F_recover) * torch.norm(F)
    # print('-', F_recover_rescale.numpy())
    FU, FD, FV= torch.svd(F_recover, some=True)
    FDnew = torch.diag(FD)
    FDnew[2, 2] = 0
    # print(FU.size(), FDnew.size(), FV.t().size())
    F_recover_sing = torch.mm(FU, torch.mm(FDnew, FV.t()))
    # F_recover_sing_rescale = F_recover_sing / torch.norm(F_recover_sing) * torch.norm(F)

    if normalize:
        F_recover_sing = torch.mm(T2.t(), torch.mm(F_recover_sing, T1))
    return F_recover_sing

def _YFX(X, Y, F, if_homo=False, clamp_at=None):
    if not if_homo:
        X = _homo(X)
        Y = _homo(Y)
    if len(X.size())==2:
        should_zeros = torch.diag(torch.matmul(torch.matmul(Y, F), X.t()))
    else:
        should_zeros = torch.diagonal(Y@F@X.transpose(1, 2), dim1=1, dim2=2)
    should_zeros_abs = should_zeros.abs()
    if clamp_at is not None:
        return torch.clamp(should_zeros_abs, max=clamp_at)
    else:
        return should_zeros_abs

def _sampson_dist(F, X, Y, if_homo=False):
    if not if_homo:
        X = utils_misc._homo(X)
        Y = utils_misc._homo(Y)
    if len(X.size())==2:
        nominator = (torch.diag(Y@F@X.t()))**2
        Fx1 = torch.mm(F, X.t())
        Fx2 = torch.mm(F.t(), Y.t())
        denom = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
    else:
        nominator = (torch.diagonal(Y@F@X.transpose(1, 2), dim1=1, dim2=2))**2
        Fx1 = torch.matmul(F, X.transpose(1, 2))
        Fx2 = torch.matmul(F.transpose(1, 2), Y.transpose(1, 2))
        denom = Fx1[:, 0]**2 + Fx1[:, 1]**2 + Fx2[:, 0]**2 + Fx2[:, 1]**2
        # print(nominator.size(), denom.size())

    errors = nominator/denom
    return errors

def _sym_epi_dist(F, X, Y, if_homo=False, clamp_at=None):
    # Actually sauqred
    if not if_homo:
        X = utils_misc._homo(X)
        Y = utils_misc._homo(Y)
    if len(X.size())==2:
        nominator = (torch.diag(Y@F@X.t()))**2
        Fx1 = torch.mm(F, X.t())
        Fx2 = torch.mm(F.t(), Y.t())
        denom_recp = 1./(Fx1[0]**2 + Fx1[1]**2) + 1./(Fx2[0]**2 + Fx2[1]**2)
    else:
        # print('-', X.detach().cpu().numpy())
        # print('-', Y.detach().cpu().numpy())
        # print('--', F.detach().cpu().numpy())
        nominator = (torch.diagonal(Y@F@X.transpose(1, 2), dim1=1, dim2=2))**2
        Fx1 = torch.matmul(F, X.transpose(1, 2))
        # print(Fx1.detach().cpu().numpy(), torch.max(Fx1), torch.sum(Fx1))
        # print(X.detach().cpu().numpy(), torch.max(X), torch.sum(X))
        Fx2 = torch.matmul(F.transpose(1, 2), Y.transpose(1, 2))
        denom_recp = 1./(Fx1[:, 0]**2 + Fx1[:, 1]**2 + 1e-10) + 1./(Fx2[:, 0]**2 + Fx2[:, 1]**2 + 1e-10)
        # print(nominator.size(), denom.size())

    errors = nominator*denom_recp
    # print('---', nominator.detach().cpu().numpy())
    # print('---------', denom_recp.detach().cpu().numpy())

    if clamp_at is not None:
        errors = torch.clamp(errors, max=clamp_at)

    return errors

def _epi_distance(F, X, Y, if_homo=False):
    # Not squared. https://arxiv.org/pdf/1706.07886.pdf
    if not if_homo:
        X = utils_misc._homo(X)
        Y = utils_misc._homo(Y)
    if len(X.size())==2:
        nominator = torch.diag(Y@F@X.t()).abs()
        Fx1 = torch.mm(F, X.t())
        Fx2 = torch.mm(F.t(), Y.t())
        denom_recp_Y_to_FX = 1./torch.sqrt(Fx1[0]**2 + Fx1[1]**2)
        denom_recp_X_to_FY = 1./torch.sqrt(Fx2[0]**2 + Fx2[1]**2)
    else:
        nominator = (torch.diagonal(Y@F@X.transpose(1, 2), dim1=1, dim2=2)).abs()
        Fx1 = torch.matmul(F, X.transpose(1, 2))
        Fx2 = torch.matmul(F.transpose(1, 2), Y.transpose(1, 2))
        denom_recp_Y_to_FX = 1./torch.sqrt(Fx1[:, 0]**2 + Fx1[:, 1]**2)
        denom_recp_X_to_FY = 1./torch.sqrt(Fx2[:, 0]**2 + Fx2[:, 1]**2)
        # print(nominator.size(), denom.size())
    dist1 = nominator*denom_recp_Y_to_FX
    dist2 = nominator*denom_recp_X_to_FY
    return (dist1+dist2)/2., dist1, dist2

def epi_distance_np(F, X, Y, if_homo=False):
    # Not squared. https://arxiv.org/pdf/1706.07886.pdf
    if not if_homo:
        X = utils_misc.homo_np(X)
        Y = utils_misc.homo_np(Y)
    if len(X.shape)==2:
        nominator = np.abs(np.diag(Y@F@X.T))
        Fx1 = F @ X.T
        Fx2 = F.T @ Y.T
        denom_recp_Y_to_FX = 1./np.sqrt(Fx1[0]**2 + Fx1[1]**2)
        denom_recp_X_to_FY = 1./np.sqrt(Fx2[0]**2 + Fx2[1]**2)
    else:
        nominator = np.abs(np.diagonal(np.transpose(Y@F@X, (1, 2)), axis=1, axis2=2))
        Fx1 = F @np.transpose(X, (1, 2))
        Fx2 = np.transpose(F, (1, 2)) @ np.transpose(Y, (1, 2))
        denom_recp_Y_to_FX = 1./np.sqrt(Fx1[:, 0]**2 + Fx1[:, 1]**2)
        denom_recp_X_to_FY = 1./np.sqrt(Fx2[:, 0]**2 + Fx2[:, 1]**2)
        # print(nominator.size(), denom.size())
    dist1 = nominator * denom_recp_Y_to_FX
    dist2 = nominator * denom_recp_X_to_FY
    dist3 = nominator * (denom_recp_Y_to_FX + denom_recp_X_to_FY)
    # return (dist1+dist2)/2., dist1, dist2
    return dist3, dist1, dist2

# def epi_distance_np_deepF(pts, F):
#     pts1 = pts2hom(pts[:,:2])
#     pts2 = pts2hom(pts[:,2:])

#     l2 = np.dot(F.T  , pts1.T)
#     l1 = np.dot(F, pts2.T)

#     dd = np.sum(l2.T*pts2,1)

#     d = np.abs(dd)*(1.0/np.sqrt(l1[0,:]**2 + l1[1,:]**2) + 1.0/np.sqrt(l2[0,:]**2 + l2[1,:]**2))

#     return d

def compute_epi_residual(pts1, pts2, F, clamp_at=0.5):
    import logging
    l1 = pts2 @ F
    l2 = pts1 @ F.permute(0,2,1)

    dd = ((pts1*l1).sum(2))
    epi = 1e-6
    d = dd.abs()*(1/(l1[:,:,:2].norm(2,2) + epi) + 1/(l2[:,:,:2].norm(2,2) + epi)) # d1+d2, no squared
    # logging.info(f'd: {d}, dd: {dd}')
    # dd = d.pow(2)

    out = torch.clamp(d, max=clamp_at)

    return out

# def compute_epi_residual(pts1, pts2, F, clamp_at=0.5):
#     l1 = pts2 @ F
#     l2 = pts1 @ F.permute(0,2,1)

#     dd = ((pts1*l1).sum(2))

#     d = dd.abs().pow(2) * ((1/(l1[:,:,:2].norm(2,2))).pow(2) + (1/(l2[:,:,:2].norm(2,2))).pow(2)) # d1**2+d2**2, no squared
#     d = torch.clamp(d, min=0.)
#     d = torch.sqrt(d)
#     # dd = d.pow(2)

#     out = torch.clamp(d, max=clamp_at)

#     return out


# def compute_epi_residual(pts1, pts2, F, clamp_at=0.5):
#     l1 = torch.bmm(pts2, F)
#     l2 = torch.bmm(pts1, F.permute(0,2,1))
#     epi = 1e-10

#     dd = ((pts1*l1).sum(2)).abs()

#     dnom = 1./(l1[:,:,0]**2 + l1[:,:,1]**2 + epi) + 1/(l2[:,:,0]**2 + l2[:,:,1]**2 + epi)   

#     out = torch.clamp(dd * torch.sqrt(dnom), max=clamp_at)

#     return out

# def compute_epi_residual_non_rob(pts1, pts2, F):
#     l1 = torch.bmm(pts2, F)
#     l2 = torch.bmm(pts1, F.permute(0,2,1))

#     dd = ((pts1*l1).sum(2))

#     d = dd.abs()*(1/(l1[:,:,:2].norm(2,2)) + 1/(l2[:,:,:2].norm(2,2)))

#     return d


def _F_to_E(F, K):
    E = torch.matmul(torch.matmul(K.t(), F), K)
    U, S, V = torch.svd(E, some=False) # https://github.com/marktao99/python/blob/da2682f8832483650b85b0be295ae7eaf179fcc5/CVP/samples/sfm.py#L139
    # print(S.numpy())
    S_110 = torch.diag(torch.tensor([1., 1., 0.], dtype=F.dtype))
    E_110 = torch.mm(U, torch.mm(S_110, V.t()))
    # print(E_110.numpy())
    return E_110

def _E_to_F(E, K):
    if len(E.size())==2:
        F = torch.matmul(torch.matmul(torch.inverse(K).t(), E), torch.inverse(K))
    else:
        F = torch.inverse(K).transpose(1, 2) @ E @ torch.inverse(K)
    return F

def E_to_F_np(E, K):
    if len(E.shape)==2:
        F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
    else:
        np.transpose(np.linalg.inv(K), (1, 2)) @ E @ np.linalg.inv(K)
    return F

def _get_M2s(E):
    # Getting 4 possible poses from E
    U, S, V = torch.svd(E)
    W = torch.tensor([[0,-1,0], [1,0,0], [0,0,1]], dtype=E.dtype, device=E.device)
    if torch.det(torch.mm(U, torch.mm(W, V.t())))<0:
        W = -W
    # print('-- delta_t_gt', delta_t_gt)

    t_recover = U[:, 2:3]/torch.norm(U[:, 2:3])
    # print('---', E.numpy())
    # t_recover_rescale = U[:, 2]/torch.norm(U[:, 2])*np.linalg.norm(t_gt) # -t_recover_rescale is also an option
    R_recover_1 = torch.mm(U, torch.mm(W, V.t()))
    R_recover_2 = torch.mm(U, torch.mm(W.t(), V.t())) # also an option
    # print('-- t_recover', t_recover.numpy())
    # print('-- R_recover_1', R_recover_1.numpy(), torch.det(R_recover_1).numpy())
    # print('-- R_recover_2', R_recover_2.numpy(), torch.det(R_recover_2).numpy())

    R2s = [R_recover_1, R_recover_2]
    t2s = [t_recover, -t_recover]
    M2s = [torch.cat((x, y), 1) for x, y in [(x,y) for x in R2s for y in t2s]]
    return R2s, t2s, M2s

def _get_M2s_batch(Es_batch):
    from batch_svd import batch_svd # https://github.com/KinglittleQ/torch-batch-svd.git
    # Getting 4 possible poses from E
    Us, Ss, Vs = batch_svd(Es_batch)
    Ws_list = []
    for U, V in zip(Us, Vs):
        W = torch.tensor([[0,-1,0], [1,0,0], [0,0,1]], dtype=Es_batch.dtype, device=Es_batch.device)
        if torch.det(torch.mm(U, torch.mm(W, V.t())))<0:
            W = -W
        Ws_list.append(W)
    Ws = torch.stack(Ws_list)

    t_recover_batch = Us[:, :, 2:3]/torch.norm(Us[:, :, 2:3], p=2, dim=1, keepdim=True)
    R_recover_1_batch = Us @ Ws @ Vs.transpose(1, 2)
    R_recover_2_batch = Us @ Ws.transpose(1, 2) @ Vs.transpose(1, 2) # also an option

    R2s_batch = [R_recover_1_batch, R_recover_2_batch]
    t2s_batch = [t_recover_batch, -t_recover_batch]
    # M2s = [torch.cat((x, y), 1) for x, y in [(x,y) for x in R2s for y in t2s]]
    return R2s_batch, t2s_batch

def _E_to_M(E_est_th, K, x1, x2, inlier_mask=None, delta_Rt_gt=None, 
            depth_thres=50., show_debug=False, show_result=True, method_name='ours'):
    if show_debug:
        print('--- Recovering pose from E...')
    count_N = x1.shape[0]
    R2s, t2s, M2s = _get_M2s(E_est_th)

    R1 = np.eye(3)
    t1 = np.zeros((3, 1))
    M1 = np.hstack((R1, t1))

    if inlier_mask is not None:
        x1 = x1[inlier_mask, :]
        x2 = x2[inlier_mask, :]
        if x1.shape[0] < 8:
            print('ERROR! Less than 8 points after inlier mask!')
            print(inlier_mask)
            return None
    # Cheirality check following OpenCV implementation: https://github.com/opencv/opencv/blob/808ba552c532408bddd5fe51784cf4209296448a/modules/calib3d/src/five-point.cpp#L513
    depth_thres = depth_thres
    cheirality_checks = []
    M2_list = []
    error_Rt = ()

    def within_mask(Z, thres_min, thres_max):
        return (Z > thres_min) & (Z < thres_max)

    for Rt_idx, M2 in enumerate(M2s):
        M2 = M2.numpy()
        R2 = M2[:, :3]
        t2 = M2[:, 3:4]
        if show_debug:
            print(M2)
            print(np.linalg.det(R2))
        X_tri_homo = cv2.triangulatePoints(np.matmul(K, M1), np.matmul(K, M2), x1.T, x2.T)
        X_tri = X_tri_homo[:3, :]/X_tri_homo[-1, :]
        # C1 = -np.matmul(R1, t1) # https://math.stackexchange.com/questions/82602/how-to-find-camera-position-and-rotation-from-a-4x4-matrix
        # cheirality1 = np.matmul(R1[2:3, :], (X_tri-C1)).reshape(-1) # https://cmsc426.github.io/sfm/
        # if show_debug:
        #     print(X_tri[-1, :])
        cheirality_mask_1 = within_mask(X_tri[-1, :], 0., depth_thres)

        X_tri_cam2 = np.matmul(R2, X_tri) + t2
        # C2 = -np.matmul(R2, t2)
        # cheirality2 = np.matmul(R2[2:3, :], (X_tri_cam3-C2)).reshape(-1)
        cheirality_mask_2 = within_mask(X_tri_cam2[-1, :], 0., depth_thres)

        cheirality_mask_12 = cheirality_mask_1 & cheirality_mask_2
        cheirality_checks.append(cheirality_mask_12)

    if show_debug:
        print([np.sum(mask) for mask in cheirality_checks])
    good_M_index, non_zero_nums = max(enumerate([np.sum(mask) for mask in cheirality_checks]), key=operator.itemgetter(1))
    
    # return values
    error_Rt = []
    Rt_cam = []
    if non_zero_nums > 0:
        # Rt_idx = cheirality_checks.index(True)
        M_inv = utils_misc.Rt_depad(np.linalg.inv(utils_misc.Rt_pad(M2s[good_M_index].numpy())))
        if show_result:
            print('The %d_th (0-based) Rt meets the Cheirality Condition! with [R|t] (camera):\n'%good_M_index, M_inv)

        if delta_Rt_gt is not None:
            R2 = M2s[good_M_index][:, :3].numpy()
            t2 = M2s[good_M_index][:, 3:4].numpy()
            # error_R = min([utils_geo.rot12_to_angle_error(R2.numpy(), delta_R_gt) for R2 in R2s])
            # error_t = min(utils_geo.vector_angle(t2, delta_t_gt), utils_geo.vector_angle(-t2, delta_t_gt))

            R2 = M_inv[:, :3]
            t2 = M_inv[:, 3:4]
            error_R = utils_geo.rot12_to_angle_error(R2, delta_Rt_gt[:3, :3]) # [RUI] Both of camera motion
            error_t = utils_geo.vector_angle(t2, delta_Rt_gt[:3, 3:4])
            if show_result:
                print('Recovered by %s (camera): The rotation error (degree) %.4f, and translation error (degree) %.4f'%(method_name, error_R, error_t))
            error_Rt = [error_R, error_t]
            Rt_cam = [R2, t2]

    else:
        # raise ValueError('ERROR! 0 of qualified [R|t] found!')
        print('ERROR! 0 of qualified [R|t] found!')
        

        # # Get rid of small angle points. @Manmo: you should discard points that are beyond a depth threshold (say, more than 100m), or which subtend a small angle between the two cameras (say, less than 5 degrees).
        # v1s = (X_tri-C1).T
        # v2s = (X_tri-C2).T
        # angles_X1_C1C2 = utils_geo.vectors_angle(v1s, v2s).reshape(-1)

        # v1s = (X_tri_cam3-C1).T
        # v2s = (X_tri_cam3-C2).T
        # angles_X2_C1C2 = utils_geo.vectors_angle(v1s, v2s).reshape(-1)

        # # angles_thres = 0.5
        # # # angles_thres = np.median(angles_X1_C1C2)
        # # angles_mask = angles_X1_C1C2 > angles_thres
        # # if show_debug:
        # #     print('!!! Good angles %d/%d with threshold %.2f'%(np.sum(angles_mask), angles_X1_C1C2.shape[0], angles_thres))

        # depth_thres = 30.
        # # print(X_tri[-1, :] > 0.)
        # # depth_mask = np.logical_and(X_tri[-1, :] > 0., X_tri[-1, :] < depth_thres).reshape(-1)
        # depth_mask = (X_tri[-1, :] < depth_thres).reshape(-1)
        # # print(angles_mask.shape)

        # # if angles_mask is not None:
        # if not np.any(depth_mask):
        #     cheirality_check = False
        #     # print('ERROR! No corres above the threshold of %.2f degrees!'%angles_thres)
        #     if show_debug:
        #         print('No depth within the threshold of 0-%.2f!'%depth_thres)
        #     # print(angles_C1C2)
        # else:
        #     # cheirality_check = np.min(cheirality1[depth_mask])>0 and np.min(cheirality2[depth_mask])>0
        #     cheirality_check = np.min(X_tri[-1, :].reshape(-1)[depth_mask])>0 and np.min(X_tri_cam3[-1, :].reshape(-1)[depth_mask])>0

        # # else:
        # #     cheirality_check = np.min(cheirality1)>0 and np.min(cheirality2)>0
        # cheirality_checks.append(cheirality_check)
        # if cheirality_check:
        #     print('-- Good M (scene):', M2)
        #     M2_list.append(M2)

        # if show_debug: # for debugging prints
        #     # print(X_tri[-1, angles_mask.reshape([-1])])
        #     # print(X_tri_cam3[-1, angles_mask.reshape([-1])])
        #     # outliers1 = cheirality1[depth_mask] < 0
        #     # print(angles_X1_C1C2[angles_mask].shape, outliers1.shape)
        #     # print(outliers1.shape, 'Outlier angles: ', angles_X1_C1C2[angles_mask][outliers1])
        #     print(X_tri[-1, :].reshape(-1))
        #     print(X_tri[-1, :].reshape(-1)[depth_mask])
        # #     # print(angles_X1_C1C2.shape, outliers1.shape)
        #     # print(angles_X1_C1C2, angles_X1_C1C2[depth_mask][outliers1])
        # #     # print(angles_X2_C1C2)
        # #     # print(X_tri[-1, :])
        # #     # print(cheirality1)
        # #     # print(cheirality2)

    # if np.sum(cheirality_checks)==1:
    #     Rt_idx = cheirality_checks.index(True)
    #     M_inv = utils_misc.Rt_depad(np.linalg.inv(utils_misc.Rt_pad(M2s[Rt_idx].numpy())))
    #     print('The %d_th Rt meets the Cheirality Condition! with [R|t] (camera):\n'%Rt_idx, M_inv)

    #     if delta_Rt_gt is not None:
    #         R2 = M2s[Rt_idx][:, :3].numpy()
    #         t2 = M2s[Rt_idx][:, 3:4].numpy()
    #         # error_R = min([utils_geo.rot12_to_angle_error(R2.numpy(), delta_R_gt) for R2 in R2s])
    #         # error_t = min(utils_geo.vector_angle(t2, delta_t_gt), utils_geo.vector_angle(-t2, delta_t_gt))

    #         R2 = M_inv[:, :3]
    #         t2 = M_inv[:, 3:4]
    #         error_R = utils_geo.rot12_to_angle_error(R2, delta_Rt_gt[:, :3])
    #         error_t = utils_geo.vector_angle(t2, delta_Rt_gt[:, 3:4])
    #         print('Recovered by %s (camera): The rotation error (degree) %.4f, and translation error (degree) %.4f'%(method_name, error_R, error_t))
    #         error_Rt = (error_R, error_t)

    #     print(M_inv)
    # else:
    #     raise ValueError('ERROR! %d of qualified [R|t] found!'%np.sum(cheirality_checks))
    #     # print('ERROR! %d of qualified [R|t] found!'%np.sum(cheirality_checks))

    return M2_list, error_Rt, Rt_cam

def _E_to_M_train(E_est_th, K, x1, x2, inlier_mask=None, delta_Rt_gt_cam=None, depth_thres=50., show_debug=False, show_result=True, method_name='ours'):
    """ convert essential matrix to pose (done Cheirality check)
    params:
        E_est_th -> torch [3, 3]: essential matrix
        K -> np [3, 3]: intrinsic matirx
        x1, x2 -> np [N, 2]: matches
    return:
        M2_list, 
        error_Rt, 
        Rt_cam -> torch [3, 4]: 
    """
    if show_debug:
        print('--- Recovering pose from E...')
    count_N = x1.shape[0]
    R2s, t2s, M2s = _get_M2s(E_est_th)

    R1 = np.eye(3)
    t1 = np.zeros((3, 1))
    M1 = np.hstack((R1, t1))

    if inlier_mask is not None:
        x1 = x1[inlier_mask, :]
        x2 = x2[inlier_mask, :]
        if x1.shape[0] < 8:
            print('ERROR! Less than 8 points after inlier mask!')
            print(inlier_mask)
            return None
    # Cheirality check following OpenCV implementation: https://github.com/opencv/opencv/blob/808ba552c532408bddd5fe51784cf4209296448a/modules/calib3d/src/five-point.cpp#L513
    depth_thres = depth_thres
    cheirality_checks = []
    M2_list = []
    error_Rt = ()

    def within_mask(Z, thres_min, thres_max):
        return (Z > thres_min) & (Z < thres_max)

    for Rt_idx, M2 in enumerate(M2s):
        M2 = M2.detach().cpu().numpy()
        R2 = M2[:, :3]
        t2 = M2[:, 3:4]
        if show_debug:
            print(M2)
            print(np.linalg.det(R2))
        X_tri_homo = cv2.triangulatePoints(np.matmul(K, M1), np.matmul(K, M2), x1.T, x2.T)
        X_tri = X_tri_homo[:3, :]/X_tri_homo[-1, :]
        # C1 = -np.matmul(R1, t1) # https://math.stackexchange.com/questions/82602/how-to-find-camera-position-and-rotation-from-a-4x4-matrix
        # cheirality1 = np.matmul(R1[2:3, :], (X_tri-C1)).reshape(-1) # https://cmsc426.github.io/sfm/
        # if show_debug:
        #     print(X_tri[-1, :])
        cheirality_mask_1 = within_mask(X_tri[-1, :], 0., depth_thres)

        X_tri_cam2 = np.matmul(R2, X_tri) + t2
        # C2 = -np.matmul(R2, t2)
        # cheirality2 = np.matmul(R2[2:3, :], (X_tri_cam3-C2)).reshape(-1)
        cheirality_mask_2 = within_mask(X_tri_cam2[-1, :], 0., depth_thres)

        cheirality_mask_12 = cheirality_mask_1 & cheirality_mask_2
        cheirality_checks.append(cheirality_mask_12)

    if show_debug:
        print([np.sum(mask) for mask in cheirality_checks])
    good_M_index, non_zero_nums = max(enumerate([np.sum(mask) for mask in cheirality_checks]), key=operator.itemgetter(1))
    if non_zero_nums > 0:
        # Rt_idx = cheirality_checks.index(True)
        # M_inv = utils_misc.Rt_depad(np.linalg.inv(utils_misc.Rt_pad(M2s[good_M_index].detach().cpu().numpy())))
        # M_inv = utils_misc.inv_Rt_np(M2s[good_M_index].detach().cpu().numpy())
        M_inv_th = utils_misc._inv_Rt(M2s[good_M_index])
        # print(M_inv, M_inv_th)
        if show_debug:
            print('The %d_th (0-based) Rt meets the Cheirality Condition! with [R|t] (camera):\n'%good_M_index, M_inv_th.detach().cpu().numpy())

        if delta_Rt_gt_cam is not None:
            # R2 = M2s[good_M_index][:, :3].numpy()
            # t2 = M2s[good_M_index][:, 3:4].numpy()
            # error_R = min([utils_geo.rot12_to_angle_error(R2.numpy(), delta_R_gt) for R2 in R2s])
            # error_t = min(utils_geo.vector_angle(t2, delta_t_gt), utils_geo.vector_angle(-t2, delta_t_gt))
            M_inv = M_inv_th.detach().cpu().numpy()
            R2 = M_inv[:, :3]
            t2 = M_inv[:, 3:4]
            error_R = utils_geo.rot12_to_angle_error(R2, delta_Rt_gt_cam[:3, :3]) # [RUI] Both of camera motion
            error_t = utils_geo.vector_angle(t2, delta_Rt_gt_cam[:3, 3:4])
            if show_result:
                print('Recovered by %s (camera): The rotation error (degree) %.4f, and translation error (degree) %.4f'%(method_name, error_R, error_t))
            error_Rt = [error_R, error_t]
        else:
            error_Rt = []
        Rt_cam = M_inv_th

    else:
        # raise ValueError('ERROR! 0 of qualified [R|t] found!')
        print('ERROR! 0 of qualified [R|t] found!')
        error_Rt = []
        Rt_cam = None

    return M2_list, error_Rt, Rt_cam



# ------ For homography ------

def _H_from_XY(X, Y):
    N = list(X.size())[0]
    A = torch.zeros(2*N, 9, dtype=torch.float32)
    A[0::2, 0:2] = X
    A[0::2, 2:3] = torch.ones(N, 1)
    A[1::2, 3:5] = X
    A[1::2, 5:6] = torch.ones(N, 1)
    A[0::2, 6:8] = X
    A[1::2, 6:8] = X
    A[:, 8:9] = torch.ones(2*N, 1)
    Y_vec = torch.reshape(Y, (2*N, 1))
    A[:, 6:7] = -A[:, 6:7] * Y_vec
    A[:, 7:8] = -A[:, 7:8] * Y_vec
    A[:, 8:9] = -A[:, 8:9] * Y_vec
    U, S, V = torch.svd(A)
    H = torch.reshape(V[:, -1], (3, 3))
    H = H / H[2, 2]
    return H

def H_from_XY_np(X, Y):
    N = X.shape[0]
    A = np.zeros((2*N, 9))
    A[0::2, 0:2] = X
    A[0::2, 2:3] = np.ones((N, 1))
    A[1::2, 3:5] = X
    A[1::2, 5:6] = np.ones((N, 1))
    A[0::2, 6:8] = X
    A[1::2, 6:8] = X
    A[:, 8:9] = np.ones((2*N, 1))
    y_vec = np.reshape(Y, (2*N, 1))
    A[:, 6:7] = -A[:, 6:7] * y_vec
    A[:, 7:8] = -A[:, 7:8] * y_vec
    A[:, 8:9] = -A[:, 8:9] * y_vec
    U, S, V = np.linalg.svd(A)
    H = np.reshape(V[-1, :], (3, 3))
    H = H / H[2, 2]
    return H

def _reproj_error_HXY(H, X, Y):
    HX = de_homo_py(torch.matmul(H, homo_py(X).t()).t())
    errors = torch.norm(Y - HX, dim=1)
    return torch.mean(errors), errors

import operator as op
from functools import reduce
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return int(numer / denom)

def _E_F_from_Rt(R_th, t_th, K_th, tensor_input=False):
    """ Better use F instead of E """
    if not tensor_input:
        K_th = torch.from_numpy(K_th).to(torch.float64)
        R_th = torch.from_numpy(R_th).to(torch.float64)
        t_th = torch.from_numpy(t_th).to(torch.float64)
    t_gt_x = utils_misc._skew_symmetric(t_th)
#     print(t_gt_x, R_th)
    E_gt_th = t_gt_x@R_th
    if len(R_th.size())==2:
        F_gt_th = torch.matmul(torch.matmul(torch.inverse(K_th).t(), E_gt_th), torch.inverse(K_th))
    else:
        F_gt_th = torch.inverse(K_th).transpose(1, 2) @ E_gt_th @ torch.inverse(K_th)
    return E_gt_th, F_gt_th

def E_F_from_Rt_np(R, t, K):
    """ Better use F instead of E """
    t_gt_x = utils_misc.skew_symmetric_np(t)
#     print(t_gt_x, R_th)
    E_gt = t_gt_x@R
    # print('E_F_from_Rt_np======', R)
    # print(R, t, K)
    if len(R.shape)==2:
        F_gt = np.linalg.inv(K).T @ E_gt @ np.linalg.inv(K)
    else:
        F_gt = np.linalg.inv(K).transpose(1, 2) @ E_gt @ np.linalg.inv(K)
    return E_gt, F_gt

def vali_with_best_M(F_gt_th, E_gt_th, x1, x2, img1_rgb_np, img2_rgb_np, kitti_two_frame_loader, delta_Rtij_inv, \
    best_N = 10, if_need_only_idx=False):
    """ Validate pose estimation with best 10 corres."""
    # Validation: use best 10 corres with smalles Sampson distance to GT F to compute E and F
    print('>>>>>>>>>>>>>>>> Check with best %d corres. ---------------'%best_N)
    # _geo_dist = _sampson_dist
    _geo_dist = _sym_epi_dist
    errors = _geo_dist(F_gt_th, torch.from_numpy(x1).to(torch.float64), torch.from_numpy(x2).to(torch.float64), False)
    sort_index = np.argsort(errors.numpy())

    best_N = best_N
    mask_index = sort_index[:best_N]
    # random.seed(10)
    # mask_index = random.sample(range(x1.shape[0]), 8)
    print('--- Best %d errors'%best_N, errors[mask_index].numpy())
    # print(errors[sort_index].numpy())

    # print('--- F GT\n', F_gt_th.numpy())

    # F_opencv, _ = cv2.findFundamentalMat(x1[mask_index, :], x2[mask_index, :], method=cv2.FM_8POINT) # based on the five-point algorithm solver in [Nister03]((1, 2) Nistér, D. An efficient solution to the five-point relative pose problem, CVPR 2003.). [SteweniusCFS](Stewénius, H., Calibrated Fivepoint solver. http://www.vis.uky.edu/~stewe/FIVEPOINT/) is also a related. 
    # F_opencv = F_gt_th.numpy()[2, 2] * F_opencv
    # print('--- F opencv\n', F_opencv)

    # # F_third = compute_fundamental_scipy(utils_misc.homo_np(x1[mask_index, :]).T, utils_misc.homo_np(x2[mask_index, :]).T)
    # # F_third = F_gt_th.numpy()[2, 2] * F_third
    # # print('--- F scipy\n', F_third)

    # # F_third, A = compute_fundamental_np(utils_misc.homo_np(x1[mask_index, :]).T, utils_misc.homo_np(x2[mask_index, :]).T)
    # # F_third = F_gt_th.numpy()[2, 2] * F_third
    # # print('--- F np\n', F_third)

    # F_est_th = _F_from_XY(torch.from_numpy(x1[mask_index, :]), torch.from_numpy(x2[mask_index, :]), W=None, show_debug=False)
    # print('--- F est (should agree with F opencv)\n', (F_est_th.numpy() / F_est_th.numpy()[2, 2] * F_gt_th.numpy()[2, 2]))
    # # print('--- F np\n', (F_np / F_np[2, 2] * F_gt_th.numpy()[2, 2]))


    # ## Check number of inliers w.r.t F_gt and thres
    # errors_estF = _sampson_dist(F_est_th, torch.from_numpy(x1).to(torch.float64), torch.from_numpy(x2).to(torch.float64), False)
    # e = np.sort(errors_estF.numpy().tolist())
    # print('--- %d/%d inliers for estimated F.'%(sum(e<DSAC_params['inlier_thresh']), len(e)))

    E_est_th = _E_from_XY(torch.from_numpy(x1[mask_index, :]), torch.from_numpy(x2[mask_index, :]), kitti_two_frame_loader.K_th, \
        W=None, show_debug=False)

    if if_need_only_idx:
        return mask_index, E_est_th.numpy()

    U,S,V = torch.svd(E_est_th)
    print('[info.Debug @vali_with_best_M] Singular values for recovered E:\n', S.numpy())

    M2_list = _E_to_M(E_est_th, kitti_two_frame_loader.K, x1, x2, delta_Rt_gt=delta_Rtij_inv, show_debug=False, method_name='Ours_best%d'%best_N)

    utils_vis.draw_corr(img1_rgb_np, img2_rgb_np, x1[mask_index, :], x2[mask_index, :], 2)

    print('GT camera matrix: (camnera)\n', delta_Rtij_inv)

    print('<<<<<<<<<<<<<<<< DONE Check with best %d corres. ---------------'%best_N)

    return mask_index, None

### ==== funcs from the Good Corr paper repo
def goodCorr_eval_nondecompose(p1s, p2s, E_hat, delta_Rtij_inv, K, scores, if_my_decomp=False):
    # Use only the top 10% in terms of score to decompose, we can probably
    # implement a better way of doing this, but this should be just fine.
    if scores is not None:
        num_top = len(scores) // 10
        num_top = max(1, num_top)
        th = np.sort(scores)[::-1][num_top] ## [RUI] Only evaluating the top 10% corres.
        mask = scores >= th

        p1s_good = p1s[mask]
        p2s_good = p2s[mask]
    else:
        p1s_good, p2s_good = p1s, p2s

    # Match types
    # E_hat = E_hat.reshape(3, 3).astype(p1s.dtype))
    if p1s_good.shape[0] >= 5:
        # Get the best E just in case we get multipl E from findEssentialMat
        # num_inlier, R, t, mask_new = cv2.recoverPose(
        #     E_hat, p1s_good, p2s_good)
        if if_my_decomp:
            M2_list, error_Rt, Rt_cam = _E_to_M(torch.from_numpy(E_hat), torch.from_numpy(p1s_good), torch.from_numpy(p2s_good), delta_Rt_gt=delta_Rtij_inv, show_debug=False, method_name='Ours_best%d'%best_N)
            if not Rt_cam:
                return None, None
            else:
                print(Rt_cam[0], Rt_cam[1])
        else:
            num_inlier, R, t, mask_new = cv2.recoverPose(E_hat, p1s_good, p2s_good, focal=K[0, 0], pp=(K[0, 2], K[1, 2]))
        try:
            R_cam, t_cam = utils_geo.invert_Rt(R, t)
            err_q = utils_geo.rot12_to_angle_error(R_cam, delta_Rtij_inv[:3, :3])
            err_t = utils_geo.vector_angle(t_cam, delta_Rtij_inv[:3, 3:4])
            # err_q, err_t = evaluate_R_t(dR, dt, R, t) # (3, 3) (3,) (3, 3) (3, 1)
        except:
            print("Failed in evaluation")
            print(R)
            print(t)
            err_q = 180.
            err_t = 90.
    else:
        err_q = 180.
        err_t = 90.
        R = np.eye(3).astype(np.float32)
        t = np.zeros((3, 1), np.float32)

    return np.hstack((R, t)), (err_q, err_t)



# def compute_fundamental_scipy(x1,x2):
#     from scipy import linalg
#     """    Computes the fundamental matrix from corresponding points 
#         (x1,x2 3*n arrays) using the 8 point algorithm.
#         Each row in the A matrix below is constructed as
#         [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] """
    
#     n = x1.shape[1]
#     if x2.shape[1] != n:
#         raise ValueError("Number of points don't match.")
    
#     # build matrix for equations
#     A = zeros((n,9))
#     for i in range(n):
#         A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
#                 x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
#                 x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
            
#     # compute linear least square solution
#     U,S,V = linalg.svd(A)
#     F = V[-1].reshape(3,3)
        
#     # constrain F
#     # make rank 2 by zeroing out last singular value
#     U,S,V = linalg.svd(F)
#     S[2] = 0
#     F = dot(U,dot(diag(S),V))
    
#     return F/F[2,2]

# def compute_fundamental_np(x1,x2):
#     """    Computes the fundamental matrix from corresponding points 
#         (x1,x2 3*n arrays) using the 8 point algorithm.
#         Each row in the A matrix below is constructed as
#         [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] """

#     n = x1.shape[1]
#     if x2.shape[1] != n:
#         raise ValueError("Number of points don't match.")
    
#     # build matrix for equations
#     A = zeros((n,9))
#     for i in range(n):
#         A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
#                 x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
#                 x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
            
#     # compute linear least square solution
#     U,S,V = np.linalg.svd(A)
#     F = V[-1].reshape(3,3)
        
#     # # constrain F
#     # # make rank 2 by zeroing out last singular value
#     # U,S,V = np.linalg.svd(F)
#     # S[2] = 0
#     # F = dot(U,dot(diag(S),V))
    
#     return F/F[2,2], A