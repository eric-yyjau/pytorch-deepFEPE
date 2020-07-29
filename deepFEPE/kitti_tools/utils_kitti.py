import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
import torch
from path import Path
from imageio import imread

import pykitti  # install using pip install pykitti
# from kitti_tools.kitti_raw_loader import read_calib_file, transform_from_rot_trans
import deepFEPE.dsac_tools.utils_misc as utils_misc
import deepFEPE.dsac_tools.utils_vis as utils_vis
import deepFEPE.dsac_tools.utils_geo as utils_geo

class KittiLoader(object):
    def __init__(self, KITTI_ROOT_PATH):
        self.KITTI_ROOT_PATH = KITTI_ROOT_PATH
        self.KITTI_PATH = KITTI_ROOT_PATH + '/raw' if 'raw' not in self.KITTI_ROOT_PATH else self.KITTI_ROOT_PATH


    def set_drive(self, date, drive, drive_path=None):
        # self.date_name = date
        # self.seq_name = seq

        # # tracklet_path = KITTI_PATH+'/%s/%s/tracklet_labels.xml'%(date_name, date_name+seq_name)
        # self.fdir_path = self.KITTI_PATH+'/%s/%s/'%(self.date_name, self.date_name+self.seq_name)
        # # if os.path.exists(tracklet_path):
        # #     print('======Tracklet Exists:', tracklet_path)
        # # else:
        # #     print('======Tracklet NOT Exists:', tracklet_path)
            
        # ## Raw Data directory information
        # path = self.fdir_path.rstrip('/')
        # basedir = path.rsplit('/',2)[0]
        # date = path.split('/')[-2]
        # drive = path.split('/')[-1].split('_')[-2]
        if drive_path is None:
            self.drive_path = self.KITTI_PATH+'/%s/%s_drive_%s_sync'%(date, date, drive)
        else:
            self.drive_path = drive_path
        self.dataset = pykitti.raw(self.KITTI_PATH, date, drive)
        self.dataset_gray = list(self.dataset.gray)
        self.dataset_rgb = list(self.dataset.rgb) 
        self.N_frames = len(self.dataset_rgb)
        if self.N_frames == 0:
            return
        ## From Rui
        # Understanding calibs: https://github.com/utiasSTARS/pykitti/blob/0e5fd7fefa7cd10bbdfb5bd131bb58481d481116/pykitti/raw.py#L150
        # cam = 'leftRGB'
        self.P_rects = {'leftRGB': self.dataset.calib.P_rect_20, 'rightRGB': self.dataset.calib.P_rect_30} # cameras def.: https://github.com/utiasSTARS/pykitti/blob/19d29b665ac4787a10306bbbbf8831181b38eb38/pykitti/odometry.py#L42
        # cam2cam = {}
        self.R_cam2rect = self.dataset.calib.R_rect_00 # [cam0] R_rect_xx: 3x3 rectifying rotation to make image planes co-planar
        P_rect = self.P_rects['leftRGB'] # P_rect_0[0-3]: 3x4 projection matrix after rectification; the reprojection matrix in MV3D
        self.velo2cam = self.dataset.calib.T_cam0_velo_unrect
        self.P_velo2im = np.dot(np.dot(P_rect, self.R_cam2rect), self.velo2cam) # 4*3
        self.im_shape = [self.dataset_gray[0][0].size[1], self.dataset_gray[0][0].size[0]] if self.N_frames!= 0 else [-1, -1]

        # print('KITTI track loaded at %s.'%self.fdir_path)

    def load_cam_poses(self):
        oxts_path = self.drive_path + '/oxts/data/*.txt'
        oxts = sorted(glob.glob(oxts_path))

        c = '02'
        scene_data = {'cid': c, 'dir': self.drive_path, 'speed': [], 'frame_id': [], 'imu_pose_matrix':[], 'rel_path': Path(self.drive_path).name + '_' + c}
        scale = None
        origin = None
        imu2velo_dict = read_calib_file(self.drive_path+'/../calib_imu_to_velo.txt')
        velo2cam_dict = read_calib_file(self.drive_path+'/../calib_velo_to_cam.txt')
        cam2cam_dict = read_calib_file(self.drive_path+'/../calib_cam_to_cam.txt')

        velo2cam_mat = transform_from_rot_trans(velo2cam_dict['R'], velo2cam_dict['T'])
        imu2velo_mat = transform_from_rot_trans(imu2velo_dict['R'], imu2velo_dict['T'])
        cam_2rect_mat = transform_from_rot_trans(cam2cam_dict['R_rect_00'], np.zeros(3))

        # self.imu2cam = self.Rtl_gt.copy() @ cam_2rect_mat @ velo2cam_mat @ imu2velo_mat
        self.imu2cam = cam_2rect_mat @ velo2cam_mat @ imu2velo_mat

        for n, f in enumerate(oxts):
            metadata = np.genfromtxt(f)
            speed = metadata[8:11]
            scene_data['speed'].append(speed)
            scene_data['frame_id'].append('{:010d}'.format(n))
            lat = metadata[0]

            if scale is None:
                scale = np.cos(lat * np.pi / 180.)
            imu_pose_matrix = pose_from_oxts_packet(metadata[:6], scale)
            # if origin is None:
            #     origin = pose_matrix

            # odo_pose = self.imu2cam @ np.linalg.inv(origin) @ pose_matrix @ np.linalg.inv(self.imu2cam)
            # # odo_pose = np.linalg.inv(origin) @ pose_matrix

            # odo_pose_Rt = odo_pose[:3]
            # R21 = odo_pose_Rt[:, :3]
            # t21 = odo_pose_Rt[:, 3:4]
            # # R12 = R21.T
            # # t12 = -np.matmul(R12, t21)

            # delta_Rtij = utils_misc.Rt_depad(np.linalg.inv(utils_misc.Rt_pad(np.hstack((R21, t21)))))
            # R12 = delta_Rtij[:, :3]
            # t12 = delta_Rtij[:, 3:4]

            # Rt12 = np.hstack((R12, t12))
            # scene_data['pose'].append(Rt12)
            scene_data['imu_pose_matrix'].append(imu_pose_matrix.copy())

        self.scene_data = scene_data
        return scene_data

    def show_demo(self):
        velo_reproj_list = []
        for i in range(self.N_frames):
            velo = list(self.dataset.velo)[i] # [N, 4]
            # project the points to the camera
            velo = velo[:, :3]
            velo_reproj = utils_misc.homo_np(velo)
            velo_reproj_list.append(velo_reproj)

            for cam_iter, cam in enumerate(['leftRGB', 'rightRGB']):
                P_rect = self.P_rects[cam] # P_rect_0[0-3]: 3x4 projection matrix after rectification; the reprojection matrix in MV3D
                P_velo2im = np.dot(np.dot(P_rect, self.R_cam2rect), self.velo2cam) # 4*3

                velo_pts_im = np.dot(P_velo2im, velo_reproj.T).T # [*, 3]
                velo_pts_im[:, :2] = velo_pts_im[:,:2] / velo_pts_im[:,2][..., np.newaxis]

                # check if in bounds
                # use minus 1 to get the exact same value as KITTI matlab code
                velo_pts_im[:, 0] = np.round(velo_pts_im[:,0]) - 1
                velo_pts_im[:, 1] = np.round(velo_pts_im[:,1]) - 1
                val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
                val_inds = val_inds & (velo_pts_im[:,0] < self.im_shape[1]) & (velo_pts_im[:,1] < self.im_shape[0])
                velo_pts_im = velo_pts_im[val_inds, :]
                
                if i == 0:
                    print('Demo: Showing left/right data (unfiltered and unrectified) of the first frame.')
                    plt.figure(figsize=(30, 8))
                    plt.imshow(self.dataset_rgb[i][cam_iter])
                    plt.scatter(velo_pts_im[:, 0].astype(np.int), velo_pts_im[:, 1].astype(np.int), s=2, c=1./velo_pts_im[:, 2])
                    plt.xlim(0, self.im_shape[1]-1)
                    plt.ylim(self.im_shape[0]-1, 0)
                    plt.title(cam)
                    plt.show()

    def get_left_right_gt(self):
        # print(self.dataset.calib.P_rect_20)
        # print(self.dataset.calib.P_rect_30)
        self.K = self.dataset.calib.P_rect_20[:3, :3]
        self.K_th = torch.from_numpy(self.K)
        self.Ml_gt = np.matmul(np.linalg.inv(self.K), self.dataset.calib.P_rect_20)
        self.Mr_gt = np.matmul(np.linalg.inv(self.K), self.dataset.calib.P_rect_30)

        tl_gt = self.Ml_gt[:, 3:4]
        tr_gt = self.Mr_gt[:, 3:4]
        Rl_gt = self.Ml_gt[:, :3]
        Rr_gt = self.Mr_gt[:, :3]
        # print('GT camera for left/right.')
        # print(Rl_gt)
        # print(Rr_gt)
        # print(tl_gt)
        # print(tr_gt)

        self.Rtl_gt = np.vstack((np.hstack((Rl_gt, tl_gt)), np.array([0., 0., 0., 1.], dtype=np.float64))) # Scene motion
        self.delta_Rtlr_gt = np.matmul(np.hstack((Rr_gt, tr_gt)), np.linalg.inv(self.Rtl_gt))
        self.delta_Rlr_gt = self.delta_Rtlr_gt[:, :3]
        self.delta_tlr_gt = self.delta_Rtlr_gt[:, 3:4]
        # print(self.delta_Rlr_gt, self.delta_tlr_gt)
        tlr_gt_x = utils_misc._skew_symmetric(torch.from_numpy(self.delta_tlr_gt).float())
        self.Elr_gt_th = torch.matmul(tlr_gt_x, torch.eye(3)).to(torch.float64)
        self.Flr_gt_th = torch.matmul(torch.matmul(torch.inverse(self.K_th).t(), self.Elr_gt_th), torch.inverse(self.K_th))

    def rectify(self, velo_reproj, im_l, im_r, visualize=False):
        val_inds_list = []

        X_homo = np.dot(np.dot(self.R_cam2rect, self.velo2cam), velo_reproj.T) # 4*N
        X_homo_rect = np.matmul(self.Rtl_gt, X_homo)
        X_rect = X_homo_rect[:3, :] / X_homo_rect[3:4, :]

        front_mask = X_rect[-1, :]>0
        X_rect = X_rect[:, front_mask]
        X_homo_rect = X_homo_rect[:, front_mask]

        # Plot with recfitied X and R, t
        x1_homo = np.matmul(self.K, np.matmul(np.hstack((np.eye(3), np.zeros((3, 1)))), X_homo_rect)).T
        x1 = x1_homo[:, 0:2]/x1_homo[:, 2:3]
        if visualize:
            plt.figure(figsize=(30, 8))
            plt.imshow(im_l)
            val_inds = utils_vis.scatter_xy(x1, x1_homo[:, 2], self.im_shape, 'Reprojection to cam 2 with rectified X and camera', new_figure=False)
        else:
            val_inds = utils_misc.within(x1[:, 0], x1[:, 1], self.im_shape[1], self.im_shape[0])
    #     print(val_inds.shape)
        val_inds_list.append(val_inds)

        x2_homo = np.matmul(self.K, np.matmul(np.hstack((self.delta_Rlr_gt, self.delta_tlr_gt)), X_homo_rect)).T
        x2 = x2_homo[:, :2]/x2_homo[:, 2:3]
        if visualize:
            plt.figure(figsize=(30, 8))
            plt.imshow(im_r)
            val_inds = utils_vis.scatter_xy(x2, x2_homo[:, 2], self.im_shape, 'Reprojection to cam 3 with rectified X and camera', new_figure=False)
        else:
            val_inds = utils_misc.within(x1[:, 0], x1[:, 1], self.im_shape[1], self.im_shape[0])
    #     print(val_inds.shape)
        val_inds_list.append(val_inds)

        # val_inds_both = val_inds_list[0] & val_inds_list[1]
        # val_idxes = [idx for idx in range(val_inds_both.shape[0]) if val_inds_both[idx]] # within indexes

        val_idxes = utils_misc.vis_masks_to_inds(val_inds_list[0], val_inds_list[1])
        return val_idxes, X_rect # list, 3*N

    def rectify_all(self, visualize=False):
        # for each frame, get the visible points on front view with identity left camera, as well as indexes of points on both left/right images
        print('Rectifying...')
        self.val_idxes_list = []
        self.X_rect_list = []
        for i in range(self.N_frames):
            print(i, self.N_frames)
            velo = list(self.dataset.velo)[i] # [N, 4]
            velo = velo[:, :3]
            velo_reproj = utils_misc.homo_np(velo)
            val_idxes, X_rect = self.rectify(velo_reproj, self.dataset_rgb[i][0], self.dataset_rgb[i][1], visualize=((i%100==0)&visualize))
            self.val_idxes_list.append(val_idxes)
            self.X_rect_list.append(X_rect)
        print('Finished rectifying all frames.')
        return self.val_idxes_list, self.X_rect_list

    def get_ij(self, i, j, visualize=False):
        """ Return frame i and j with point cloud from i, and relative camera pose [R|t] """
        # Rt0 = self.scene_data['pose'][0] # Identity, or = utils_misc.identity_Rt()
        # Rti = self.scene_data['pose'][i]
        # Rtj = self.scene_data['pose'][j]

        # print('Rti', Rti)
        # print('Rti', Rtj)

        X_rect_i = self.X_rect_list[i]

        np.set_printoptions(precision=8, suppress=True)\

        # delta_Rtij = utils_misc.Rt_depad(np.linalg.inv(utils_misc.Rt_pad(Rti)) @ utils_misc.Rt_pad(Rtj))
        odo_pose = self.imu2cam @ np.linalg.inv(self.scene_data['imu_pose_matrix'][i]) @ self.scene_data['imu_pose_matrix'][j] @ np.linalg.inv(self.imu2cam) # camera motion
        # delta_Rtij = utils_misc.Rt_depad(np.linalg.inv(odo_pose)) # scene motion;  [RUI] Cam 0

        print(self.imu2cam @ np.linalg.inv(self.scene_data['imu_pose_matrix'][j]) @ self.scene_data['imu_pose_matrix'][i] @ np.linalg.inv(self.imu2cam))

        delta_Rtij = utils_misc.Rt_depad(self.Rtl_gt @ np.linalg.inv(odo_pose) @ np.linalg.inv(self.Rtl_gt)) # [RUI] Cam 2
        val_inds_i, _ = utils_vis.reproj_and_scatter(utils_misc.identity_Rt(), X_rect_i.T, self.dataset_rgb[i][0], self, visualize=visualize, title_appendix='frame %d (left)'%i, set_lim=True)
        val_inds_j, _ = utils_vis.reproj_and_scatter(delta_Rtij, X_rect_i.T, self.dataset_rgb[j][0], self, visualize=visualize, title_appendix='frame %d (left)'%j, set_lim=True)
        X_rect_j = self.X_rect_list[j]
        # val_inds_j = utils_vis.reproj_and_scatter(Rt0, X_rect_j, self.dataset_rgb[j][0], self, visualize=visualize)  
        val_idxes = utils_misc.vis_masks_to_inds(val_inds_i, val_inds_j)
        X_rect_i_vis = X_rect_i[:, val_idxes]

        delta_Rtij_inv = utils_misc.Rt_depad(odo_pose) # camera motion

        # print(delta_Rtij_inv)

        angle_R = utils_geo.rot12_to_angle_error(np.eye(3), delta_Rtij_inv[:, :3])
        angle_t = utils_geo.vector_angle(np.array([[0.], [0.], [1.]]), delta_Rtij_inv[:, 3:4])

        print('>>>>>>>>>>>>>>>> Between frame %d and %d: \nThe rotation angle (degree) %.4f, and translation angle (degree) %.4f'%(i, j, angle_R, angle_t))


        return X_rect_i, X_rect_i_vis, delta_Rtij, delta_Rtij_inv, self.dataset_rgb[i][0], self.dataset_rgb[j][0]

    def get_i_lr(self, i, visualize=False):
        """ Return frame i left and right with point cloud from i, and relative camera pose [R|t] """
        Rt0 = self.scene_data['pose'][0] # Identity, or = utils_misc.identity_Rt()
        # Rti = self.scene_data['pose'][i]
        # Rtj = self.scene_data['pose'][j]

        X_rect_i = self.X_rect_list[i]
        # delta_Rtij = utils_misc.Rt_depad(np.linalg.inv(utils_misc.Rt_pad(Rti)) @ utils_misc.Rt_pad(Rtj))
        delta_Rtij = self.delta_Rtlr_gt

        val_inds_i_l = utils_vis.reproj_and_scatter(Rt0, X_rect_i, self.dataset_rgb[i][0], self, visualize=visualize, title_appendix='frame %d (left)'%i)
        val_inds_i_r = utils_vis.reproj_and_scatter(delta_Rtij, X_rect_i, self.dataset_rgb[i][1], self, visualize=visualize, title_appendix='frame %d (right)'%i)
        val_idxes = utils_misc.vis_masks_to_inds(val_inds_i_l, val_inds_i_r)

        X_rect_i_vis = X_rect_i[:, val_idxes]

        delta_Rtij_inv = utils_misc.Rt_depad(np.linalg.inv(utils_misc.Rt_pad(delta_Rtij))) # camera motion

        return X_rect_i, X_rect_i_vis, delta_Rtij, delta_Rtij_inv, self.dataset_rgb[i][0], self.dataset_rgb[i][1]


def pose_from_oxts_packet(metadata, scale):

    lat, lon, alt, roll, pitch, yaw = metadata
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    Taken from https://github.com/utiasSTARS/pykitti
    """

    er = 6378137.  # earth radius (approx.) in meters
    # Use a Mercator projection to get the translation vector
    ty = lat * np.pi * er / 180.

    tx = scale * lon * np.pi * er / 180.
    # ty = scale * er * \
    #     np.log(np.tan((90. + lat) * np.pi / 360.))
    tz = alt
    t = np.array([tx, ty, tz]).reshape(-1,1)

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(roll)
    Ry = roty(pitch)
    Rz = rotz(yaw)
    R = Rz.dot(Ry.dot(Rx))
    return transform_from_rot_trans(R, t)


def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data

def rectify(velo_homo, calibs):
        val_inds_list = []

        X_homo = np.dot(np.dot(calibs['cam_2rect'], calibs['velo2cam']), velo_homo.T) # 4*N
        X_cam0 = utils_misc.de_homo_np(X_homo.T).copy()
        X_rect_homo = np.matmul(calibs['Rtl_gt'], X_homo)
        X_rect = X_rect_homo[:3, :] / X_rect_homo[3:4, :]

        front_mask = X_rect[-1, :]>0
        X_rect = X_rect[:, front_mask]
        X_rect_homo = X_rect_homo[:, front_mask]
        X_cam0 = X_cam0[front_mask, :]

        # Plot with recfitied X and R, t
        x1_homo = np.matmul(calibs['K'], np.matmul(np.hstack((np.eye(3), np.zeros((3, 1)))), X_rect_homo)).T
        x1 = x1_homo[:, 0:2]/x1_homo[:, 2:3]
        val_inds = utils_misc.within(x1[:, 0], x1[:, 1], calibs['im_shape'][1]-1, calibs['im_shape'][0]-1)
        val_inds_list.append(val_inds)

        # x2_homo = np.matmul(calibs['K'], np.matmul(np.hstack((calibs['delta_Rlr_gt'], calibs['delta_tlr_gt'])), X_homo_rect)).T
        # x2 = x2_homo[:, :2]/x2_homo[:, 2:3]
        # val_inds = utils_misc.within(x1[:, 0], x1[:, 1], calibs['im_shape'][1]-1, calibs['im_shape'][0]-1)
        # val_inds_list.append(val_inds)

        # val_inds_both = val_inds_list[0] & val_inds_list[1]
        # val_idxes = [idx for idx in range(val_inds_both.shape[0]) if val_inds_both[idx]] # within indexes

        val_idxes = utils_misc.vis_masks_to_inds(val_inds_list[0], val_inds_list[0])
        return val_idxes, X_rect.T, X_cam0

def load_velo_scan(velo_filename):
    # https://github.com/charlesq34/frustum-pointnets/blob/889c277144a33818ddf73c4665753975f9397fc4/kitti/kitti_util.py#L270
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan

def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])

def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def scale_intrinsics(mat, sx, sy):
    print('--scale_intrinsics')
    assert mat.shape==(3, 3)
    out = np.copy(mat)
    out[0,0] *= sx
    out[0,2] *= sx
    out[1,1] *= sy
    out[1,2] *= sy
    return out

def scale_P(P, sx, sy, if_print=False):
    if if_print:
        print(f'---scale_P - sx={sx}, sy={sy}')
    assert P.shape==(3, 4)
    out = np.copy(P)
    out[0] *= sx
    out[1] *= sy
    return out

def load_as_float(path):
    return np.array(imread(path)).astype(np.float32)

def load_as_array(path, dtype=None):
    array = np.load(path)
    if dtype is not None:
        return array.astype(dtype)
    else:
        return array

def load_sift(dump_dir, frame_nb, ext):
    sift_file = dump_dir/'sift_{}'.format(frame_nb) + ext
    sift_array = load_as_array(sift_file, np.float32)
    sift_kp = sift_array[:, :2] # [N, 2]
    sift_des = sift_array[:, 2:] # [N, 128]
    return sift_kp, sift_des

def load_SP(dump_dir, frame_nb, ext):
    SP_file = dump_dir/'SP_{}'.format(frame_nb) + ext
    SP_array = load_as_array(SP_file, np.float32)
    SP_kp = SP_array[:, :3] # [N, 2]
    SP_des = SP_array[:, 3:] # [N, 128]
    return SP_kp, SP_des