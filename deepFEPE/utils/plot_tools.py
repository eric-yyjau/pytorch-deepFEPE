import argparse
import time
import csv
import yaml
import os
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tensorboardX import SummaryWriter
import cv2
import matplotlib.pyplot as plt


class plot_results(object):
    def __init__(self, frame_list=[100], mode='base'):
        #         frame_list = [0, 100, 200, 300]
        # frame_list = [100, 700, 1200]
        # frame_list = [100]
        self.frame_list = frame_list
        print(f"mode = {mode}")
        self.get_image_names(mode=mode)
        pass

    def get_image_names(self, mode='base'):
        frame_list = self.frame_list
        plot_folder = "plots/"
        image_name = None
        if mode == 'base':
            prefix = ["Si-Df-k", "Sp-Df-fp-end-k"]
            plot_name = "mask_conf_"  # 'corr_all_'
            #         image_name = [f"{plot_folder}{plot_name}{prefix}{i:06}_{(i+1):06}.png" for i in frame_list]
        elif mode == 'good' or mode == 'bad':
            prefix = [f"Si-Df-fp-k_{mode}", f"Sp-Df-fp-end-k_{mode}"]
            plot_name = "mask_conf_" # "mask_conf_"  # 'corr_all_'
        elif mode == 'freeze':
            print(f"freeze!")
            iter_list = [0, 400, 1000]
            prefix_base = "Sp-Df-f-end-k-freezeDf"
            plot_name = 'corr_all_random_'  # 'corr_all_', "mask_conf_" "epi_dist_all_" "corr_all_random_"
            print(f"plot_name: {plot_name}")
            # prefix = [f'{prefix_base}_{iter/1000}k_' for iter in iter_list]  # 'Sp-Df-fp-end-k'
            prefix = [f'{prefix_base}_s{frame_list[0]}_{iter/1000}k' for iter in iter_list]  # 'Sp-Df-fp-end-k'
            image_name = [f"{plot_folder}{plot_name}{p}.png" for p in prefix]
            # prefix = f'Sp-Df-f-end-k-freezeDf_s{j}_{iter/1000}k'
        # image_name = [
        #     f"{plot_folder}{plot_name}{pre}{i:06}_{(i+1):06}.png"
        #     for i in frame_list
        #     for pre in prefix
        # ]
        if image_name is None:
            image_name = [
                f"{plot_folder}{plot_name}{pre}_{i}.png"
                for i in frame_list
                for pre in prefix
            ]
        self.prefix = prefix
        self.image_name = image_name
        self.image_data = []
        self.plot_name = plot_name
        print(image_name)        

    def __len__(self):
        return len(self.image_name)

    def read_images(self):
        image_data = []
        image_name = self.image_name
        for i, file in enumerate(image_name):
            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_data.append(img)
            print(f"read {i}: {file}")
        #     plt.imshow(img)
        #     plt.show()
        self.image_data = image_data
        pass

    def plot_images(
        self, row=2, col=2, col_labels=["Baseline - Si-Df-fp", "Ours - Sp-Df-fp-end"],
        save=True,
        figsize=(48,12),
        ext='pdf'
    ):
        ## create subgraph for combinations
        #         row, col = 2, 2
        img_num = row * col
        assert self.__len__() >= img_num
        image_data = self.image_data

        f, axarr = plt.subplots(row, col, figsize=figsize)
        #         f, axarr = plt.subplots(row, col, figsize=(48, 12))

        axarr = axarr.reshape(-1, col)
        for i in range(img_num):
            print(f"axarr: {axarr.shape}, i= {i}")
            axarr[int(i / col), int(i % col)].imshow(image_data[i])
            axarr[int(i / col), int(i % col)].axis("off")
        #     axarr[i/2,i%2].imshow(imaget(_datas[1])
        #     axarr[1,0].imshow(image_datas[2])
        #     axarr[1,1].imshow(image_datas[3])

        for ax, col_name in zip(axarr[0], col_labels):
            ax.set_title(col_name, fontsize=figsize[0])

        f.tight_layout()
        #         f.suptitle(f'{self.prefix}', fontsize=12)
        savefile = f"{self.plot_name}_{str('_').join(self.prefix)}_{str('_').join([str(f) for f in self.frame_list])}"
        if save:
            if ext == 'pdf':
                file = f"plots/{savefile}.pdf"
                plt.savefig(file, bbox_inches="tight")
            else:
                file = f"plots/{savefile}.png"
                plt.savefig(file, dpi=300, bbox_inches="tight")
            logging.info(f"save image: {savefile}")
            print(f"save image: {file}")
        else:
            print(f"not saved!!")
        #         logging.info(f"save image: {file}")
        plt.show()

if __name__ == "__main__":
    plot_helper = plot_class()
    plot_helper.read_images()
    # plot_helper.plot_images(row=3,col=2)
    plot_helper.plot_images(row=1,col=2)


# class plot_class(object):
#     def __init__(self):
# #         frame_list = [0, 100, 200, 300]
#         frame_list = [100, 700, 1200]
# #         frame_list = [100]
#         prefix = ['Si-Df-k', 'Sp-Df-fp-end-k']
#         plot_folder = 'plots/'
#         plot_name = 'mask_conf_' # 'corr_all_'
# #         image_name = [f"{plot_folder}{plot_name}{prefix}{i:06}_{(i+1):06}.png" for i in frame_list]
#         image_name = [f"{plot_folder}{plot_name}{pre}{i:06}_{(i+1):06}.png" for i in frame_list for pre in prefix ]
#         self.frame_list = frame_list
#         self.prefix = prefix
#         self.image_name = image_name
#         self.image_data = []
#         print(image_name)        
#         pass
#     def __len__(self):
#         return len(self.image_name)
    
#     def read_images(self):
#         image_data = []
#         image_name = self.image_name
#         for i, file in enumerate(image_name):
#             img = cv2.imread(file)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             image_data.append(img)
#             print(f"read {i}: {file}")
#         #     plt.imshow(img)
#         #     plt.show()
#         self.image_data = image_data
#         pass
    
#     def plot_images(self, row=2, col=2, col_labels=['Baseline - Si-Df-fp', 'Ours - Sp-Df-fp-end']):
#         ## create subgraph for combinations
# #         row, col = 2, 2
#         img_num = row*col
#         assert self.__len__() >= img_num
#         image_data = self.image_data
        
#         f, axarr = plt.subplots(row, col, figsize=(48, 12))
# #         f, axarr = plt.subplots(row, col, figsize=(48, 12))
        
#         axarr = axarr.reshape(-1, col)
#         for i in range(img_num):
#             print(f'axarr: {axarr.shape}, i= {i}')
#             axarr[int(i/col),int(i%col)].imshow(image_data[i])
#             axarr[int(i/col),int(i%col)].axis('off')
#         #     axarr[i/2,i%2].imshow(imaget(_datas[1])
#         #     axarr[1,0].imshow(image_datas[2])
#         #     axarr[1,1].imshow(image_datas[3])

        
#         for ax, col_name in zip(axarr[0], col_labels):
#             ax.set_title(col_name)
        
#         f.tight_layout()
# #         f.suptitle(f'{self.prefix}', fontsize=12)
#         savefile = f"{str('_').join(self.prefix)}_{str('_').join([str(f) for f in self.frame_list])}"
#         file = f"plots/{savefile}.png"
# #         logging.info(f"save image: {file}")
#         print(f"save image: {file}")
#         plt.show()    



# def plot_imgs(imgs, titles=None, cmap='brg', ylabel='', normalize=False, ax=None, dpi=100):
#     n = len(imgs)
#     if not isinstance(cmap, list):
#         cmap = [cmap]*n
#     if ax is None:
#         fig, ax = plt.subplots(1, n, figsize=(6*n, 6), dpi=dpi)
#         if n == 1:
#             ax = [ax]
#     else:
#         if not isinstance(ax, list):
#             ax = [ax]
#         assert len(ax) == len(imgs)
#     for i in range(n):
#         if imgs[i].shape[-1] == 3:
#             imgs[i] = imgs[i][..., ::-1]  # BGR to RGB
#         ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmap[i]),
#                      vmin=None if normalize else 0,
#                      vmax=None if normalize else 1)
#         if titles:
#             ax[i].set_title(titles[i])
#         ax[i].get_yaxis().set_ticks([])
#         ax[i].get_xaxis().set_ticks([])
#         for spine in ax[i].spines.values():  # remove frame
#             spine.set_visible(False)
#     ax[0].set_ylabel(ylabel)
#     plt.tight_layout()


# # from utils.draw import img_overlap
# def img_overlap(img_r, img_g, img_gray):  # img_b repeat
#     img = np.concatenate((img_gray, img_gray, img_gray), axis=0)
#     img[0, :, :] += img_r[0, :, :]
#     img[1, :, :] += img_g[0, :, :]
#     img[img > 1] = 1
#     img[img < 0] = 0
#     return img

# def draw_keypoints(img, corners, color=(0, 255, 0), radius=3, s=3):
#     '''

#     :param img:
#         image:
#         numpy [H, W]
#     :param corners:
#         Points
#         numpy [N, 2]
#     :param color:
#     :param radius:
#     :param s:
#     :return:
#         overlaying image
#         numpy [H, W]
#     '''
#     img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
#     for c in np.stack(corners).T:
#         # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
#         cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, color, thickness=-1)
#     return img

# # def draw_keypoints(img, corners, color=(0, 255, 0), radius=3, s=3):
# #     '''

# #     :param img:
# #         np (H, W)
# #     :param corners:
# #         np (3, N)
# #     :param color:
# #     :param radius:
# #     :param s:
# #     :return:
# #     '''
# #     img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
# #     for c in np.stack(corners).T:
# #         # cv2.circle(img, tuple(s * np.flip(c, 0)), radius, color, thickness=-1)
# #         cv2.circle(img, tuple((s*c[:2]).astype(int)), radius, color, thickness=-1)
# #     return img

# def draw_matches(rgb1, rgb2, match_pairs, filename='matches.png', show=False):
#     '''

#     :param rgb1:
#         image1
#         numpy (H, W)
#     :param rgb2:
#         image2
#         numpy (H, W)
#     :param match_pairs:
#         numpy (keypoiny1 x, keypoint1 y, keypoint2 x, keypoint 2 y)
#     :return:
#         None
#     '''
#     from matplotlib import pyplot as plt

#     h1, w1 = rgb1.shape[:2]
#     h2, w2 = rgb2.shape[:2]
#     canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=rgb1.dtype)
#     canvas[:h1, :w1] = rgb1[:,:,np.newaxis]
#     canvas[:h2, w1:] = rgb2[:,:,np.newaxis]
#     # fig = plt.figure(frameon=False)
#     fig = plt.imshow(canvas)

#     xs = match_pairs[:, [0, 2]]
#     xs[:, 1] += w1
#     ys = match_pairs[:, [1, 3]]

#     alpha = 1
#     sf = 5
#     lw = 0.5
#     # markersize = 1
#     markersize = 2

#     plt.plot(
#         xs.T, ys.T,
#         alpha=alpha,
#         linestyle="-",
#         linewidth=lw,
#         aa=False,
#         marker='o',
#         markersize=markersize,
#         fillstyle='none',
#         color=[0.0, 0.8, 0.0],
#     );
#     plt.tight_layout()
#     plt.savefig(filename, dpi=300, bbox_inches='tight')
#     print('#Matches = {}'.format(len(match_pairs)))
#     if show:
#         plt.show()

# # from utils.draw import draw_matches_cv
# def draw_matches_cv(data):
#     keypoints1 = [cv2.KeyPoint(p[1], p[0], 1) for p in data['keypoints1']]
#     keypoints2 = [cv2.KeyPoint(p[1], p[0], 1) for p in data['keypoints2']]
#     inliers = data['inliers'].astype(bool)
#     matches = np.array(data['matches'])[inliers].tolist()
#     def to3dim(img):
#         if img.ndim == 2:
#             img = img[:, :, np.newaxis]
#         return img
#     img1 = to3dim(data['image1'])
#     img2 = to3dim(data['image2'])
#     img1 = np.concatenate([img1, img1, img1], axis=2)
#     img2 = np.concatenate([img2, img2, img2], axis=2)
#     return cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches,
#                            None, matchColor=(0,255,0), singlePointColor=(0, 0, 255))


# def drawBox(points, img, offset=np.array([0,0]), color=(0,255,0)):
# #     print("origin", points)
#     offset = offset[::-1]
#     points = points + offset
#     points = points.astype(int)
#     for i in range(len(points)):
#         img = img + cv2.line(np.zeros_like(img),tuple(points[-1+i]), tuple(points[i]), color,5)
#     return img
