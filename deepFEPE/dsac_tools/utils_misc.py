import torch
import numpy as np
from itertools import cycle
import cv2

cycol = cycle('bgrcmk')

def _gaussian_dist(xs, mean, var):
    prob = torch.exp(-(xs-mean)**2 / (2*var))
    return prob

def within(x, y, xlim, ylim):
    val_inds = (x >= 0) & (y >= 0)
    val_inds = val_inds & (x <= xlim) & (y <= ylim)
    return val_inds

def identity_Rt(dtype=np.float32):
    return np.hstack((np.eye(3, dtype=dtype), np.zeros((3, 1), dtype=dtype)))

def _skew_symmetric(v): # v: [3, 1] or [batch_size, 3, 1]
    if len(v.size())==2:
        zero = torch.zeros_like(v[0, 0])
        M = torch.stack([
            zero, -v[2, 0], v[1, 0],
            v[2, 0], zero, -v[0, 0],
            -v[1, 0], v[0, 0], zero,
        ], dim=0)
        return M.view(3, 3)
    else:
        zero = torch.zeros_like(v[:, 0, 0])
        M = torch.stack([
            zero, -v[:, 2, 0], v[:, 1, 0],
            v[:, 2, 0], zero, -v[:, 0, 0],
            -v[:, 1, 0], v[:, 0, 0], zero,
        ], dim=1)
        return M.view(-1, 3, 3)

def skew_symmetric_np(v): # v: [3, 1] or [batch_size, 3, 1]
    if len(v.shape)==2:
        zero = np.zeros_like(v[0, 0])
        M = np.stack([
            zero, -v[2, 0], v[1, 0],
            v[2, 0], zero, -v[0, 0],
            -v[1, 0], v[0, 0], zero,
        ], axis=0)
        return M.reshape(3, 3)
    else:
        zero = np.zeros_like(v[:, 0, 0])
        M = np.stack([
            zero, -v[:, 2, 0], v[:, 1, 0],
            v[:, 2, 0], zero, -v[:, 0, 0],
            -v[:, 1, 0], v[:, 0, 0], zero,
        ], axis=1)
        return M.reshape(-1, 3, 3)
    
def _homo(x):
    # input: x [N, 2] or [batch_size, N, 2]
    # output: x_homo [N, 3]  or [batch_size, N, 3]
    assert len(x.size()) in [2, 3]
    print(f"x: {x.size()[0]}, {x.size()[1]}, {x.dtype}, {x.device}")
    if len(x.size())==2:
        ones = torch.ones(x.size()[0], 1, dtype=x.dtype, device=x.device)
        x_homo = torch.cat((x, ones), 1)
    elif len(x.size())==3:
        ones = torch.ones(x.size()[0], x.size()[1], 1, dtype=x.dtype, device=x.device)
        x_homo = torch.cat((x, ones), 2)
    return x_homo

def _de_homo(x_homo):
    # input: x_homo [N, 3] or [batch_size, N, 3]
    # output: x [N, 2] or [batch_size, N, 2]
    assert len(x_homo.size()) in [2, 3]
    epi = 1e-10
    if len(x_homo.size())==2:
        x = x_homo[:, :-1]/((x_homo[:, -1]+epi).unsqueeze(-1))
    else:
        x = x_homo[:, :, :-1]/((x_homo[:, :, -1]+epi).unsqueeze(-1))
    return x

def homo_np(x):
    # input: x [N, D]
    # output: x_homo [N, D+1]
    N = x.shape[0]
    x_homo = np.hstack((x, np.ones((N, 1), dtype=x.dtype)))
    return x_homo

def de_homo_np(x_homo):
    # input: x_homo [N, D]
    # output: x [N, D-1]
    assert x_homo.shape[1] in [3, 4]
    N = x_homo.shape[0]
    epi = 1e-10
    x = x_homo[:, :-1]/np.expand_dims(x_homo[:, -1]+epi, -1)
    return x

def Rt_pad(Rt):
    # Padding 3*4 [R|t] to 4*4 [[R|t], [0, 1]]
    assert Rt.shape==(3, 4)
    return np.vstack((Rt, np.array([[0., 0., 0., 1.]], dtype=Rt.dtype)))

# def _Rt_pad(Rt):
#     # Padding 3*4 [R|t] to 4*4 [[R|t], [0, 1]]
#     assert Rt.size()==(3, 4)
#     cat_tensor = torch.tensor([[0., 0., 0., 1.]], dtype=Rt.dtype)
#     return torch.cat((Rt, ))

def inv_Rt_np(Rt):
    assert Rt.shape==(3, 4)
    R1 = Rt[:, :3]
    t1 = Rt[:, 3:4]
    R2 = R1.T
    t2 = -R1.T @ t1
    return np.hstack((R2, t2))

def _inv_Rt(Rt):
    assert Rt.size()==(3, 4)
    R1 = Rt[:, :3]
    t1 = Rt[:, 3:4]
    R2 = R1.t()
    t2 = -R1.t() @ t1
    return torch.cat((R2, t2), 1)

def Rt_depad(Rt01):
    # dePadding 4*4 [[R|t], [0, 1]] to 3*4 [R|t]
    assert Rt01.shape==(4, 4)
    return Rt01[:3, :]

def vis_masks_to_inds(mask1, mask2):
    val_inds_both = mask1 & mask2
    val_idxes = [idx for idx in range(val_inds_both.shape[0]) if val_inds_both[idx]] # within indexes
    return val_idxes

def normalize_Rt_to_1(Rt):
    assert Rt.shape==(4, 4) or Rt.shape==(3, 4)
    if Rt.shape==(4, 4):
        Rt = Rt[:3, :]
    return np.hstack((Rt[:, :3], Rt[:, 3:4]/(Rt[2, 3]+1e-10)))

def crop_or_pad_choice(in_num_points, out_num_points, shuffle=False):
    # Adapted from https://github.com/haosulab/frustum_pointnet/blob/635c938f18b9ec1de2de717491fb217df84d2d93/fpointnet/data/datasets/utils.py
    """Crop or pad point cloud to a fixed number; return the indexes
    Args:
        points (np.ndarray): point cloud. (n, d)
        num_points (int): the number of output points
        shuffle (bool): whether to shuffle the order
    Returns:
        np.ndarray: output point cloud
        np.ndarray: index to choose input points
    """
    if shuffle:
        choice = np.random.permutation(in_num_points)
    else:
        choice = np.arange(in_num_points)
    assert out_num_points > 0, 'out_num_points = %d must be positive int!'%out_num_points
    if in_num_points >= out_num_points:
        choice = choice[:out_num_points]
    else:
        num_pad = out_num_points - in_num_points
        pad = np.random.choice(choice, num_pad, replace=True)
        choice = np.concatenate([choice, pad])
    return choice

def get_virt_x1x2_grid(im_shape):
    step = 0.1
    sz1 = im_shape
    sz2 = im_shape
    xx, yy = np.meshgrid(np.arange(0, 1 , step), np.arange(0, 1, step))
    num_pts_full = len(xx.flatten())
    pts1_virt_b = np.float32(np.vstack((sz1[1]*xx.flatten(),sz1[0]*yy.flatten())).T)
    pts2_virt_b = np.float32(np.vstack((sz2[1]*xx.flatten(),sz2[0]*yy.flatten())).T)
    return pts1_virt_b, pts2_virt_b

def get_virt_x1x2_np(im_shape, F_gt, K, pts1_virt_b, pts2_virt_b): ##  [RUI] TODO!!!!! Convert into seq loader!
    ## s.t. SHOULD BE ALL ZEROS: losses = utils_F.compute_epi_residual(pts1_virt_ori, pts2_virt_ori, F_gts, loss_params['clamp_at'])
    ## Reproject by minimizing distance to groundtruth epipolar lines
    pts1_virt, pts2_virt = cv2.correctMatches(F_gt, np.expand_dims(pts2_virt_b, 0), np.expand_dims(pts1_virt_b, 0))
    pts1_virt[np.isnan(pts1_virt)] = 0.
    pts2_virt[np.isnan(pts2_virt)] = 0.

    # nan1 = np.logical_and(
    #         np.logical_not(np.isnan(pts1_virt[:,:,0])),
    #         np.logical_not(np.isnan(pts1_virt[:,:,1])))
    # nan2 = np.logical_and(
    #         np.logical_not(np.isnan(pts2_virt[:,:,0])),
    #         np.logical_not(np.isnan(pts2_virt[:,:,1])))
    # _, midx = np.where(np.logical_and(nan1, nan2))
    # good_pts = len(midx)
    # while good_pts < num_pts_full:
    #     midx = np.hstack((midx, midx[:(num_pts_full-good_pts)]))
    #     good_pts = len(midx)
    # midx = midx[:num_pts_full]
    # pts1_virt = pts1_virt[:,midx]
    # pts2_virt = pts2_virt[:,midx]

    pts1_virt = homo_np(pts1_virt[0])
    pts2_virt = homo_np(pts2_virt[0])
    pts1_virt_normalized = (np.linalg.inv(K) @ pts1_virt.T).T
    pts2_virt_normalized = (np.linalg.inv(K) @ pts1_virt.T).T
    return pts1_virt_normalized, pts2_virt_normalized, pts1_virt, pts2_virt

def get_virt_x1x2(im_shape, F_gt, K, pts1_virt_b=None, pts2_virt_b=None): ##  [RUI] TODO!!!!! Convert into seq loader!
    ## s.t. SHOULD BE ALL ZEROS: losses = utils_F.compute_epi_residual(pts1_virt_ori, pts2_virt_ori, F_gts, loss_params['clamp_at'])
    if pts1_virt_b is None and pts2_virt_b is None:
        pts1_virt_b, pts2_virt_b = get_virt_x1x2_grid(im_shape)
    ## Reproject by minimizing distance to groundtruth epipolar lines
    pts1_virt, pts2_virt = cv2.correctMatches(F_gt, np.expand_dims(pts2_virt_b, 0), np.expand_dims(pts1_virt_b, 0))
    pts1_virt[np.isnan(pts1_virt)] = 0.
    pts2_virt[np.isnan(pts2_virt)] = 0.

    # nan1 = np.logical_and(
    #         np.logical_not(np.isnan(pts1_virt[:,:,0])),
    #         np.logical_not(np.isnan(pts1_virt[:,:,1])))
    # nan2 = np.logical_and(
    #         np.logical_not(np.isnan(pts2_virt[:,:,0])),
    #         np.logical_not(np.isnan(pts2_virt[:,:,1])))
    # _, midx = np.where(np.logical_and(nan1, nan2))
    # good_pts = len(midx)
    # while good_pts < num_pts_full:
    #     midx = np.hstack((midx, midx[:(num_pts_full-good_pts)]))
    #     good_pts = len(midx)
    # midx = midx[:num_pts_full]
    # pts1_virt = pts1_virt[:,midx]
    # pts2_virt = pts2_virt[:,midx]

    pts1_virt = utils_misc.homo_np(pts1_virt[0])
    pts2_virt = utils_misc.homo_np(pts2_virt[0])
    pts1_virt_normalized = (np.linalg.inv(K) @ pts1_virt.T).T
    pts2_virt_normalized = (np.linalg.inv(K) @ pts1_virt.T).T
    return torch.from_numpy(pts1_virt_normalized).float(), torch.from_numpy(pts2_virt_normalized).float(), \
        torch.from_numpy(pts1_virt).float(), torch.from_numpy(pts2_virt).float()