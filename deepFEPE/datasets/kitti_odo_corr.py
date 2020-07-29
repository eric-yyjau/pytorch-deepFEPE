"""
# main data loader
Organized and documented by You-Yi on 07/14/2020.

Edition history:
    

Authors:
    You-Yi Jau, Rui Zhu

"""

import numpy as np
from path import Path
import os, sys
import random
import cv2
from tqdm import tqdm
import imageio
from skimage.transform import resize

import torch
import torch.utils.data as data

from deepFEPE.utils.tools import dict_update
from deepFEPE.kitti_tools.utils_kitti import load_as_float, load_as_array
from deepFEPE.kitti_tools.utils_kitti import scale_P
import deepFEPE.dsac_tools.utils_misc as utils_misc
import deepFEPE.dsac_tools.utils_F as utils_F
import deepFEPE.dsac_tools.utils_vis as utils_vis
import deepFEPE.dsac_tools.utils_geo as utils_geo
from deepFEPE.utils.logging import *

# from datasets.kitti_tools.utils_good import *


class KittiCorrOdo(data.Dataset):
    default_config = {
        # 'labels': None,
        "cache_in_memory": True,
        # 'validation_size': 100,
        # 'truncate': None,
        # 'preprocessing': {
        # 'resize': [375*0.5, 1242*0.5]
        # 'resize_ratio': 0.5
        # },
    }

    def __init__(self, export=False, transform=None, task="train", seed=0, **config):
        torch.set_default_tensor_type(torch.FloatTensor)
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed

        self.config = self.default_config
        self.config = dict_update(self.config, config)

        self.root = Path(self.config["data"]["dump_root"])
        assert task in ["train", "val", "test"]  # test is used for testing
        frame_list_path = self.root / f"{task}.txt"
        self.task = task
        self.frames = [
            [self.root / frame[:-8], frame[-7:-1]] for frame in open(frame_list_path)
        ]  # odo: Path(root/04_02) 0
        print(f"frames: {self.frames[0]}")
        cam_ids = [frame[0][-2:] for frame in self.frames]
        assert len(set(cam_ids)) == 1
        self.cam_id = cam_ids[0]
        print(f"cam_id: {self.cam_id}")
        assert self.cam_id in ["00", "02", "_1", "_5"]
        # self.transform = transform
        self.sequence_length = self.config["data"]["sequence_length"]
        assert self.sequence_length == 2, "Sorry self.sequence_length!=2 not supported!"
        self.delta_ij = self.config["data"]["delta_ij"]
        self.image_size = self.config["data"]["image"]["size"]
        self.get_X = self.config["data"]["read_what"]["with_X"]
        self.get_pose = self.config["data"]["read_what"]["with_pose"]
        self.get_qt = self.config["data"]["read_what"]["with_qt"]
        self.get_sift = self.config["data"]["read_what"]["with_sift"]
        self.load_npy = not (self.config["data"]["read_params"]["use_h5"])
        self.img_gamma = self.config["data"]["read_what"].get("with_imgs_gamma", None)
        self.ext = ".npy" if self.load_npy else ".h5"
        self.bf = cv2.BFMatcher()

        self.sizerHW = self.image_size.copy()
        if self.config["data"]["preprocessing"]["resize"]:
            self.sizerHW = self.config["data"]["preprocessing"]["resize"]
            logging.info(
                "===[Data]=== Resizing to [H%d, W%d]"
                % (self.sizerHW[0], self.sizerHW[1])
            )

        logging.info(f"use size: {self.sizerHW} for vertual points")
        self.pts1_virt_b, self.pts2_virt_b = utils_misc.get_virt_x1x2_grid(
            self.sizerHW
        )

        self.crawl_folders(self.sequence_length)

    def crawl_folders(self, sequence_length):
        logging.info("Crawling folders for %d frames..." % len(self.frames))
        sequence_set = []
        # demi_length = (sequence_length-1)//2
        # demi_length = sequence_length-1
        # shifts = list(range(-demi_length, demi_length + 1))
        # shifts.pop(demi_length)
        # max_idx = (sequence_length-1)*self.delta_ij
        Ks = {}
        poseses = {}
        Rt_cam2_gts = {}
        scenes = list(set([frame[0] for frame in self.frames]))
        for scene in tqdm(scenes):
            # intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            Ks[scene] = (
                load_as_array(scene / "cam.npy").astype(np.float32).reshape((3, 3))
            )
            # imu_pose_matrixs = np.genfromtxt(scene/'imu_pose_matrixs.txt').astype(np.float64).reshape(-1, 4, 4)
            if self.load_npy:
                poseses[scene] = (
                    load_as_array(scene / "poses.npy")
                    .astype(np.float32)
                    .reshape(-1, 3, 4)
                )
            else:
                # imu_pose_matrixs = loadh5(scene/'imu_pose_matrixs.h5')['pose'].astype(np.float64).reshape(-1, 4, 4)
                pass
            Rt_cam2_gts[scene] = load_as_array(scene / "Rt_cam2_gt.npy")  # [4, 4]

        for frame in tqdm(self.frames):
            scene = frame[0]
            frame_id = frame[1]
            frame_num = int(frame_id)
            # print(f"get sample: {frame_nusm}")

            if self.sequence_length == 2:
                # search_name = scene/'ij_idx_{}-{}_good_ij{}'.format(frame_num, frame_num+self.delta_ij, self.ext)
                search_name = scene / "ij_match_quality_{}-{}_good{}".format(
                    frame_num, frame_num + self.delta_ij, self.ext
                )
                if not os.path.isfile(search_name):
                    logging.warning(
                        "File do not exist: %s. Skipped frame %s-%s."
                        % (search_name, scene, frame_id)
                    )
                    continue

            K = Ks[scene]
            #### scale K

            poses = poseses[scene]
            Rt_cam2_gt = Rt_cam2_gts[scene]

            def get_sample_i(i):
                img_file = scene / "%s.jpg" % i
                X_cam0_file = scene / "X_cam0_%s" % i + self.ext
                X_cam2_file = scene / "X_cam2_%s" % i + self.ext
                sift_file = scene / "sift_%s" % i + self.ext
                assert os.path.isfile(img_file) and os.path.isfile(
                    sift_file
                ), f"Some file is missing found for {str(scene)}-{str(i)}!"
                if self.get_X:
                    assert os.path.isfile(X_cam0_file) and os.path.isfile(
                        X_cam2_file
                    ), f"Some file is missing found for {str(scene)}-{str(i)}!"
                return img_file, X_cam0_file, X_cam2_file, sift_file

            sample = {
                "K_ori": K,
                # "K_inv": np.linalg.inv(K),
                "scene_name": self.task + scene.name,
                "scene": scene,
                "imgs": [],
                "cam_poses": [],
                "relative_scene_poses": [],
                "X_cam0_files": [],
                "X_cam2_files": [],
                "sift_files": [],
                "frame_ids": [],
                "ids": [],
                "Rt_cam2_gt": Rt_cam2_gt,
            }

            for k in range(0, sequence_length):
                j = k * self.delta_ij + frame_num
                img_file_j, X_cam0_file_j, X_cam2_file_j, sift_file_j = get_sample_i(
                    "%06d" % j
                )
                sample["imgs"].append(img_file_j)
                if self.get_pose:
                    sample["cam_poses"].append(
                        np.linalg.inv(utils_misc.Rt_pad(poses[j]))
                    )  # [3, 4], absolute pose read from GT file.
                    if k == 0:
                        sample["relative_scene_poses"].append(
                            utils_misc.identity_Rt(dtype=np.float32)
                        )
                    else:
                        # print(frame_num, j)
                        # if frame_num < 5: logging.info(f"frame {frame_num}: {poses[frame_num]}, Rt_cam2_gt: {Rt_cam2_gt}")
                        relative_scene_pose = np.linalg.inv(
                            utils_misc.Rt_pad(poses[j])
                        ) @ utils_misc.Rt_pad(poses[frame_num])
                        if self.cam_id == "02":
                            relative_scene_pose = (
                                Rt_cam2_gt
                                @ relative_scene_pose
                                @ np.linalg.inv(Rt_cam2_gt)
                            )
                        sample["relative_scene_poses"].append(
                            relative_scene_pose
                        )  # [4, 4]
                if self.get_X:
                    sample["X_cam0_files"].append(X_cam0_file_j)  # [3, N]
                    sample["X_cam2_files"].append(X_cam2_file_j)  # [3, N]
                if self.get_sift:
                    sample["sift_files"].append(sift_file_j)  # [N, 256+2]
                sample["frame_ids"].append(Path(img_file_j).name.replace(".jpg", ""))
                sample["ids"].append(j)
            sequence_set.append(sample)
        # random.shuffle(sequence_set)  ### why shuffle the samples... just use pytorch dataset shuffle
        # print(self.seed)
        # test_list = [1, 2, 4, 1, 4, 1, 5, 65, 66, 22]
        # random.shuffle(test_list)
        # print(test_list)
        self.samples = sequence_set

    def __getitem__(self, index):
        input = {}
        sample = self.samples[index]
        if (
            self.config["data"]["batch_size"] == 1
            or self.config["model"]["if_img_feat"]
            or self.config["model"]["if_SP"]
        ):
            # imgs = [load_as_float(img) for img in sample['imgs']]
            imgs = []
            imgs_grey = []
            zoom_xys = []
            for img_file in sample['imgs']:
                image_rgb, zoom_xy = load_and_resize_img(img_file, self.sizerHW, False)
                # img_totype = lambda x: x.astype('float32') / 255.0
                img_totype = lambda x: x.astype('float32')
                # image_rgb = np.array(image_rgb, dtype=np.uint8)  # cv2.error: OpenCV(3.4.2) /io/opencv/modules/imgproc/src/color.hpp:253: error: (-215:Assertion failed)
                imgs.append(img_totype(image_rgb)) # uint8!!!
                # print(f"image_rgb: {image_rgb.dtype}")

                img_g = img_totype(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) ) 
                ### enable gamma correction
                if self.img_gamma is not None:
                    import skimage
                    gamma = self.img_gamma
                    # logging.info(f"gamma correction: {self.img_gamma}")
                    # gamma = np.random.random_sample(1)*(gamma[1]-gamma[0]) + gamma[0]
                    # img_g = skimage.exposure.adjust_gamma(img_g, gamma=gamma[0], gain=1)

                    img_g = skimage.exposure.adjust_gamma(img_g, gamma=gamma, gain=1)
                    # logging.info(f"gamma: {gamma}, img_g: {img_g.shape}")

                    input.update({"gamma": gamma})

                imgs_grey.append(img_g)
                zoom_xys.append(zoom_xy)
            input.update({"imgs": imgs, "imgs_grey": imgs_grey, "img_zoom_xy": zoom_xy})
            zoom_xy = zoom_xys[-1] #### assume images are in the same size
        else:
            zoom_xy = (self.sizerHW[1] / self.image_size[1], self.sizerHW[0] / self.image_size[0])




        ### rescale K, create K_inv, E, F
        def add_scaled_K(sample, zoom_xy=[1,1]):
            """
            # scale calibration matrix based on img_zoom ratio. Add to the dict
            """
            if zoom_xy[0] != 1 or zoom_xy[1] != 1:
                logging.debug(f"note: scaled_K with zoom_xy = {zoom_xy}")
            P_rect_ori = np.concatenate((sample['K_ori'], [[0], [0], [0]]), axis=1).astype(np.float32)
            P_rect_scale = scale_P(
                P_rect_ori, zoom_xy[0], zoom_xy[1]
            )
            K = P_rect_scale[:, :3]
            sample.update({
                'K': K,
                "K_inv": np.linalg.inv(K),
            })
            logging.debug(f"K_ori: {sample['K_ori']}, type: {sample['K_ori'].dtype}, K: {sample['K']}, type: {K.dtype}")
            return sample

        def get_E_F(sample):
            """
            # add essential and fundamental matrix based on K.
            # *** must use the updated K!!!
            """
            relative_scene_pose = sample['relative_scene_poses'][1]
            if self.get_pose and self.sequence_length == 2:
                sample["E"], sample["F"] = utils_F.E_F_from_Rt_np(
                    relative_scene_pose[:3, :3],
                    relative_scene_pose[:3, 3:4],
                    sample["K"],
                )
            return sample

        def scale_points(points, zoom_xy, loop_length=2):
            """
            # iteratively scale x, y, x, y, ...
            """
            for i in range(loop_length):
                points[:, i] = points[:, i]*zoom_xy[i%2]
            return points
            pass

        sample = add_scaled_K(sample, zoom_xy=zoom_xy)
        sample = get_E_F(sample)

        update_list = ['K_ori', 'K', 'K_inv', 'E', 'F', 'scene_name', 'frame_ids', 'Rt_cam2_gt']
        for e in update_list:
            input.update({e: sample[e]})

        # input.update(
        #     {
        #         "scene_name": sample["scene_name"],
        #         "frame_ids": sample["frame_ids"],
        #     }
        # )


        # if self.transform is not None:
        #     imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
        #     tgt_img = imgs[0]
        #     ref_imgs = imgs[1:]
        # else:

        ids = sample["ids"]
        get_flags = {}

        ## 4k it/s
        if self.sequence_length == 2:
            # dump_ij_idx_file_name = sample['scene']/'ij_idx_{}-{}'.format(ids[0], ids[1])
            dump_ij_match_quality_file_name = sample[
                "scene"
            ] / "ij_match_quality_{}-{}".format(ids[0], ids[1])
            if self.load_npy:
                # dump_ij_idx_files = [dump_ij_idx_file_name+'_all_ij.npy', dump_ij_idx_file_name+'_good_ij.npy']
                # dump_ij_quality_file = dump_ij_idx_file_name+'_quality_good_ij.npy'
                dump_ij_match_quality_files = [
                    dump_ij_match_quality_file_name + "_all.npy",
                    dump_ij_match_quality_file_name + "_good.npy",
                ]
            else:
                # dump_ij_idx_file = dump_ij_idx_file_name+'.h5'
                pass

        if self.get_X or self.config["model"]["if_lidar_corres"]:
            if self.load_npy:
                X_cam0s = (
                    [
                        load_as_array(X_file, np.float32)
                        for X_file in sample["X_cam0_files"]
                    ]
                    if self.get_X
                    else [-1] * self.sequence_length
                )
                X_cam2s = (
                    [
                        load_as_array(X_file, np.float32)
                        for X_file in sample["X_cam2_files"]
                    ]
                    if self.get_X
                    else [-1] * self.sequence_length
                )
            else:
                # Xs = [loadh5(X_file)['X_rect_vis'] for X_file in sample['X_files']] if self.get_X else [-1]*self.sequence_length
                logging.error("Not loading if_lidar_corres!")
                pass
            # get_flags['Xs'] = self.get_X
            if self.config["data"]["batch_size"] == 1:
                input.update({"X_cam0s": X_cam0s, "X_cam2s": X_cam2s})

        if self.get_sift:
            # if not self.config["model"]["if_SP"] and self.img_gamma is not None:
            #     sift_num=2000
            #     self.if_BF_matcher = if_BF_matcher
            #     self.sift = cv2.xfeatures2d.SIFT_create(
            #         nfeatures=self.sift_num, contrastThreshold=1e-5
            #     )
            #     kp, des = self.sift.detectAndCompute(
            #         img_ori, None
            #     )  ## IMPORTANT: normalize these points
            #     x_all = np.array([p.pt for p in kp])
            #     # print(zoom_xy)
            #     x_all = (x_all * np.array([[zoom_xy[0], zoom_xy[1]]])).astype(np.float32)
            #     # print(x_all.shape, np.amax(x_all, axis=0), np.amin(x_all, axis=0))
            #     if x_all.shape[0] != self.sift_num:
            #         choice = crop_or_pad_choice(x_all.shape[0], self.sift_num, shuffle=True)
            #         x_all = x_all[choice]
            #         des = des[choice]
            #     sample["sift_kp"] = x_all
            #     sample["sift_des"] = des
            #     ## do matchin
            #     all_ij, good_ij, quality_good, quality_all = get_sift_match_idx_pair(
            #         sift_matcher, sift_des_ii.copy(), sift_des_jj.copy()
            #     )
            #     ## redo sift
            #     pass
            # elif self.load_npy:

            if self.load_npy:
                if self.config["data"]["read_what"]["with_sift_des"]:
                    sift_arrays = [
                        load_as_array(sift_file, np.float32)
                        for sift_file in sample["sift_files"]
                    ]
                    sift_kps = [
                        sift_array[:, :2] for sift_array in sift_arrays
                    ]  # list of [Ni, 2]

                    ### rescale sift kps
                    sift_kps = scale_points(sift_kps, zoom_xy)

                    sift_deses = [
                        sift_array[:, 2:] for sift_array in sift_arrays
                    ]  # list of [Ni, 128]
                    input.update({"sift_kps": sift_kps, "sift_deses": sift_deses})

                # In Good Corr, they construct a batch with the min number of points in the batch: https://github.com/vcg-uvic/learned-correspondence-release/blob/035dda508915ff5e1641e4c5c3de639deb80038a/network.py#L398
                # In Frustum Pointnet, they sample or pad:
                # [1] https://github.com/charlesq34/frustum-pointnets/blob/2ffdd345e1fce4775ecb508d207e0ad465bcca80/models/model_util.py#L52
                # [2] https://github.com/haosulab/frustum_pointnet/blob/635c938f18b9ec1de2de717491fb217df84d2d93/fpointnet/data/datasets/utils.py

                if self.sequence_length == 2:
                    # ij_idxes = []
                    match_qualitys = []
                    # for dump_ij_idx_file in dump_ij_idx_files:
                    #     if os.path.isfile(dump_ij_idx_file):
                    #         ij_idxes.append(load_as_array(dump_ij_idx_file))
                    #         get_flags['have_matches'] = True
                    #     else:
                    #         logging.warning('NOT Find '+dump_ij_idx_file)
                    #         get_flags['have_matches'] = False
                    for dump_ij_match_quality_file in dump_ij_match_quality_files:
                        if os.path.isfile(dump_ij_match_quality_file):
                            match_qualitys.append(
                                load_as_array(dump_ij_match_quality_file).astype(np.float32)
                            )
                            get_flags["have_matches"] = True
                        else:
                            logging.warning("NOT Find " + dump_ij_match_quality_file)
                            get_flags["have_matches"] = False

                    if get_flags["have_matches"]:
                        # matches_all = np.hstack((sift_kps[0][ij_idxes[0][:, 0], :], sift_kps[1][ij_idxes[0][:, 1], :]))
                        matches_all = match_qualitys[0][:, :4]
                        matches_all = scale_points(matches_all, zoom_xy, loop_length=4)
                        # print('--1', matches_all.dtype, matches_all.shape)
                        choice_all = utils_misc.crop_or_pad_choice(
                            matches_all.shape[0], 2000, shuffle=True
                        )
                        matches_all_padded = matches_all[choice_all]
                        # matches_good = np.hstack((sift_kps[0][ij_idxes[1][:, 0], :], sift_kps[1][ij_idxes[1][:, 1], :]))
                        matches_good = match_qualitys[1][:, :4]
                        matches_good = scale_points(matches_good, zoom_xy, loop_length=4)

                        choice_good = utils_misc.crop_or_pad_choice(
                            matches_good.shape[0],
                            self.config["data"]["good_num"],
                            shuffle=True,
                        )
                        matches_good_padded = matches_good[choice_good]
                        input.update(
                            {
                                "matches_all": matches_all_padded,
                                "matches_good": matches_good_padded,
                                "matches_good_unique_nums": min(
                                    matches_good.shape[0],
                                    self.config["data"]["good_num"],
                                ),
                                "matches_all_unique_nums": np.unique(matches_all, axis=0).shape[0],
                            }
                        )
                        if self.config["data"]["batch_size"] == 1:
                            input.update({"matches_good_ori": matches_good})

                    if (
                        self.config["data"]["read_what"]["with_quality"]
                        and get_flags["have_matches"]
                    ):
                        quality_good = match_qualitys[1][:, 4:]
                        # print('--2', quality_good.dtype, quality_good.shape)
                        # if os.path.isfile(dump_ij_quality_file):
                        #     quality_good = load_as_array(dump_ij_quality_file, dtype=np.float32)
                        #     get_flags['have_quality'] = True
                        # else:
                        #     logging.warning('NOT Find '+dump_ij_quality_file)
                        #     get_flags['have_quality'] = False

                        # if get_flags['have_quality'] and get_flags['have_matches']:
                        quality_good_padded = quality_good[choice_good]
                        quality_good_padded[:, 0] = (
                            quality_good_padded[:, 0] / 300.0
                        )  # Some scaling

                        quality_all_padded = quality_good_padded
                        input.update(
                            {
                                "quality_good": quality_good_padded,
                                "quality_all": quality_all_padded,
                            }
                        )

                    if self.config["data"]["read_what"]["with_sift_des"]:
                        des_good = np.hstack(
                            (
                                sift_deses[0][ij_idxes[1][:, 0], :],
                                sift_deses[1][ij_idxes[1][:, 1], :],
                            )
                        )
                        des_good_padded = des_good[choice_good]
                        input.update({"des_good": des_good_padded})
            else:
                pass

        if self.get_pose:
            input.update({"relative_scene_poses": sample["relative_scene_poses"]})
            input.update({"cam_poses": sample["cam_poses"]})
            if self.sequence_length == 2:
                input.update({"E": sample["E"], "F": sample["F"]})
                sample["pts1_virt_normalized"], sample["pts2_virt_normalized"], sample[
                    "pts1_virt"
                ], sample["pts2_virt"] = utils_misc.get_virt_x1x2_np(
                    self.image_size,
                    sample["F"],
                    sample["K"],
                    self.pts1_virt_b,
                    self.pts2_virt_b,
                )
                input.update(
                    {
                        "pts1_virt_normalized": sample["pts1_virt_normalized"],
                        "pts2_virt_normalized": sample["pts2_virt_normalized"],
                        "pts1_virt": sample["pts1_virt"],
                        "pts2_virt": sample["pts2_virt"],
                    }
                )

                if self.get_qt:
                    Rt_cam = np.linalg.inv(input["relative_scene_poses"][1])
                    R_cam = Rt_cam[:3, :3]
                    t_cam = Rt_cam[:3, 3:4]
                    q_cam = utils_geo.R_to_q_np(R_cam)
                    Rt_scene = input["relative_scene_poses"][1]
                    R_scene = Rt_scene[:3, :3]
                    t_scene = Rt_scene[:3, 3:4]
                    q_scene = utils_geo.R_to_q_np(R_scene)
                    # print(q.shape)
                    # R_re = utils_geo.q_to_R_np(q)
                    # print(R)
                    # print(R_re)
                    input.update(
                        {
                            "q_cam": q_cam,
                            "t_cam": t_cam,
                            "q_scene": q_scene,
                            "t_scene": t_scene,
                        }
                    )

                if self.get_X or self.config["model"]["if_lidar_corres"]:
                    param_list = [input["K"], self.image_size]
                    Xj_cam2 = X_cam2s[1]
                    choice_all = utils_misc.crop_or_pad_choice(
                        Xj_cam2.shape[0], 200, shuffle=True
                    )
                    sample_Xj = Xj_cam2[choice_all]
                    delta_Rtij_camid = np.linalg.inv(input["relative_scene_poses"][1])

                    _, xi = utils_vis.reproj_and_scatter(
                        utils_misc.Rt_depad(delta_Rtij_camid),
                        sample_Xj,
                        None,
                        visualize=False,
                        title_appendix="j to i",
                        param_list=param_list,
                        debug=False,
                    )
                    _, xj = utils_vis.reproj_and_scatter(
                        utils_misc.identity_Rt(),
                        sample_Xj,
                        None,
                        visualize=False,
                        title_appendix="j to j",
                        param_list=param_list,
                        debug=False,
                    )
                    input.update(
                        {
                            "pts1_velo": utils_misc.homo_np(xi),
                            "pts2_velo": utils_misc.homo_np(xj),
                        }
                    )

        input.update({"get_flags": get_flags})
        # print(f"relative_scene_poses: {input['relative_scene_poses']}")
        return input

    def __len__(self):
        return len(self.samples)


def load_and_resize_img(img_file, sizerHW, show_zoom_info=False):
    if not Path(img_file).isfile():
        logging.error("Image %s not found!" % img_file)
        return
    # img_ori = imageio.imread(img_file)
    img_ori = cv2.imread(img_file)
    img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    if [sizerHW[0], sizerHW[1]] == [img_ori.shape[0], img_ori.shape[1]]: # H, W
        return img_ori, (1., 1.)
    else:
        zoom_y = sizerHW[0] / img_ori.shape[0]
        zoom_x = sizerHW[1] / img_ori.shape[1]
        if show_zoom_info and (zoom_y != 1 or zoom_x != 0):
            logging.info(

                "[%s] Zooming the image (H%d, W%d) with zoom_yH=%f, zoom_xW=%f to (H%d, W%d)."
                % (
                    img_file,
                    img_ori.shape[0],
                    img_ori.shape[1],
                    zoom_y,
                    zoom_x,
                    sizerHW[0],
                    sizerHW[1],
                )
            )
        # img = scipy.misc.imresize(img_ori, (sizerHW[0], sizerHW[1]))
        img = cv2.resize(img_ori, (sizerHW[1], sizerHW[0]))
        return img, (zoom_x, zoom_y)


if __name__ == "__main__":
    pass

