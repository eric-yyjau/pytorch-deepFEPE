import cv2
import matplotlib.pyplot as plt
import numpy as np
import deepFEPE.dsac_tools.utils_misc as utils_misc

def drawlines(img1,img2,lines,pts1,pts2, colors=None, width=1):
    ''' OpenCV function for img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape[:2]
#     img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
#     img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    colors_random = []
    for idx, (r,pt1,pt2) in enumerate(zip(lines,pts1,pts2)):
        if colors is not None:
            color = colors[idx]
        else:
            color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,width)
        img1 = cv2.circle(img1,tuple(pt1),2,color,2)
        img2 = cv2.circle(img2,tuple(pt2),2,color,2)
        colors_random.append(color)
    return img1,img2,colors_random

def scatter_xy(xy, c, im_shape, title='', new_figure=True, s=2, 
                cmap='rainbow', set_lim=True, if_show=True, zorder=2):
    if new_figure:
        plt.figure(figsize=(60, 8))
    # plt.scatter(xy[:, 0], xy[:, 1], s=s, c=c, marker='o', cmap=cmap, zorder=zorder)
    plt.scatter(xy[:, 0], xy[:, 1], s=s, facecolors='none', linewidth=4, edgecolors=c, marker='o', cmap=cmap, zorder=zorder)
    # plt.scatter(xy[:, 0], xy[:, 1], s=s, c=c, marker='o', cmap=cmap, zorder=zorder)
    # plt.colorbar()
    if set_lim:
        plt.xlim(0, im_shape[1]-1)
        plt.ylim(im_shape[0]-1, 0)
    plt.title(title)
    if if_show: plt.show()
    val_inds = utils_misc.within(xy[:, 0], xy[:, 1], im_shape[1], im_shape[0])
    return val_inds

def show_kp(img, x, scale=1):
    plt.figure(figsize=(30*scale, 8*scale))
    plt.imshow(img)
    plt.scatter(x[:, 0], x[:, 1], s=10, marker='o', c='y')
    plt.show()

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def draw_corr(im1, im2, x1, x2, linewidth=2, 
                new_figure=True, title='', color='g', if_show=True,
                zorder=1):
    # im1 = img1_rgb
    # im2 = img2_rgb
    # x1 = x1_sample
    # x2 = x2_sample
    im_shape = im1.shape
    assert im1.shape == im2.shape, 'Shape mismatch between im1 and im2! @draw_corr()'
    x2_copy = x2.copy()
    x2_copy[:, 0] = x2_copy[:, 0] + im_shape[1]
    im12 = np.hstack((im1, im2))

    if new_figure: 
        plt.figure(figsize=(60, 8))
    plt.imshow(im12, cmap=None if len(im12.shape)==3 else plt.get_cmap('gray'))
    plt.plot(np.vstack((x1[:, 0], x2_copy[:, 0])), np.vstack((x1[:, 1], x2_copy[:, 1])), 
                marker='', linewidth=linewidth, color=color, zorder=zorder)
    if title!='':
        plt.title(title)
    if if_show: 
        plt.show()

def draw_corr_widths(im1, im2, x1, x2, linewidth, title='', rescale=True, scale=1.):
    # im1 = img1_rgb
    # im2 = img2_rgb
    # x1 = x1_sample
    # x2 = x2_sample
    im_shape = im1.shape
    assert im1.shape == im2.shape, 'Shape mismatch between im1 and im2! @draw_corr()'
    x2_copy = x2.copy()
    x2_copy[:, 0] = x2_copy[:, 0] + im_shape[1]
    im12 = np.hstack((im1, im2))

    plt.figure(figsize=(60, 8))
    plt.imshow(im12, cmap=None if len(im12.shape)==3 else plt.get_cmap('gray'))
    for i in range(x1.shape[0]):
        if rescale:
            width = 5 if linewidth[i]<2 else 10
        else:
            width = linewidth[i]*scale
        plt.plot(np.vstack((x1[i, 0], x2_copy[i, 0])), np.vstack((x1[i, 1], x2_copy[i, 1])), linewidth=width, marker='o', markersize=8)
    plt.title(title, {'fontsize':40})
    plt.show()

def draw_corr_widths_and_epi(F_gt, im1, im2, x1, x2, linewidth, title='', rescale=True, scale=1.):
    # im1 = img1_rgb
    # im2 = img2_rgb
    # x1 = x1_sample
    # x2 = x2_sample
    im_shape = im1.shape
    assert im1.shape == im2.shape, 'Shape mismatch between im1 and im2! @draw_corr()'
    x2_copy = x2.copy()
    x2_copy[:, 0] = x2_copy[:, 0] + im_shape[1]

    # lines2 = cv2.computeCorrespondEpilines(x1.reshape(-1,1,2).astype(int), 1,F_gt)
    # lines2 = lines2.reshape(-1,3)
    # im2, _, colors = drawlines(np.array(im2).copy(), np.array(im1).copy(), lines2, x2.astype(int), x1.astype(int), width=2)

    im12 = np.hstack((im1, im2))

    plt.figure(figsize=(60, 8))
    plt.imshow(im12)
    for i in range(x1.shape[0]):
        if rescale:
            width = 5 if linewidth[i]<2 else 10
        else:
            width = linewidth[i]*scale
        p = plt.plot(np.vstack((x1[i, 0], x2_copy[i, 0])), np.vstack((x1[i, 1], x2_copy[i, 1])), linewidth=width, marker='o', markersize=8)

        print(p[0].get_color())
        N_points = x1.shape[0]
        x1_homo = utils_misc.homo_np(x1)
        x2_homo = utils_misc.homo_np(x2)
        right_P = np.matmul(F_gt, x1_homo.T)
        right_epipolar_x = np.tile(np.array([[0], [1]]), N_points) * im_shape[1]
        # Using the eqn of line: ax+by+c=0; y = (-c-ax)/b, http://ai.stanford.edu/~mitul/cs223b/draw_epipolar.m
        right_epipolar_y = (-right_P[2:3, :] - right_P[0:1, :] * right_epipolar_x) / right_P[1:2, :]

        colors = np.random.rand(x2.shape[0])
        # plt.figure(figsize=(30, 8))
        # plt.subplot(121)
        # plt.imshow(img1_rgb)
        # plt.scatter(x1[:, 0], x1[:, 1], s=50, c=colors, edgecolors='w')
        # plt.subplot(122)
        # # plt.figure(figsize=(30, 8))
        # plt.imshow(img2_rgb)
        plt.plot(right_epipolar_x + im_shape[1], right_epipolar_y)
        # plt.scatter(x2[:, 0], x2[:, 1], s=50, c=colors, edgecolors='w')
    plt.xlim(0, im_shape[1]*2-1)
    plt.ylim(im_shape[0]-1, 0)
    # plt.show()

    plt.title(title, {'fontsize':40})
    plt.show()


def reproj_and_scatter(Rt, X_rect, im_rgb, kitti_two_frame_loader=None, visualize=True, title_appendix='', param_list=[], set_lim=False, debug=True, s=10):
    if kitti_two_frame_loader is None:
        if debug:
            print('Reading from input list of param_list=[K, im_shape].')
        K = param_list[0]
        im_shape = param_list[1]
    else:
        K = kitti_two_frame_loader.K
        im_shape = kitti_two_frame_loader.im_shape

    x1_homo = np.matmul(K, np.matmul(Rt, utils_misc.homo_np(X_rect).T)).T
    x1 = x1_homo[:, 0:2]/x1_homo[:, 2:3]
    if visualize:
        plt.figure(figsize=(30, 8))
        cmap = None if len(np.array(im_rgb).shape)==3 else plt.get_cmap('gray')
        plt.imshow(im_rgb, cmap=cmap)
        val_inds = scatter_xy(x1, x1_homo[:, 2], im_shape, 'Reprojection to cam 2 with rectified X and camera_'+title_appendix, new_figure=False,set_lim=set_lim, s=s)
    else:
        val_inds = utils_misc.within(x1[:, 0], x1[:, 1], im_shape[1], im_shape[0])
    return val_inds, x1

def show_epipolar_opencv(x1, x2, img1_rgb, img2_rgb, F_gt, colors=None):
    lines2 = cv2.computeCorrespondEpilines(x1.reshape(-1,1,2).astype(int), 1,F_gt)
    lines2 = lines2.reshape(-1,3)
    img3,img4, colors = drawlines(np.array(img2_rgb).copy(), np.array(img1_rgb).copy(), lines2, x2.astype(int), x1.astype(int), colors=colors)
    # plt.figure(figsize=(30, 8))
    # # plt.subplot(121)
    # plt.imshow(img4)
    # plt.subplot(122),
    plt.figure(figsize=(30, 8))
    plt.imshow(img3)
    plt.show()
    return colors


def show_epipolar_rui(x1, x2, img1_rgb, img2_rgb, F_gt, im_shape):
    N_points = x1.shape[0]
    x1_homo = utils_misc.homo_np(x1)
    x2_homo = utils_misc.homo_np(x2)
    right_P = np.matmul(F_gt, x1_homo.T)
    right_epipolar_x = np.tile(np.array([[0], [1]]), N_points) * im_shape[1]
    # Using the eqn of line: ax+by+c=0; y = (-c-ax)/b, http://ai.stanford.edu/~mitul/cs223b/draw_epipolar.m
    right_epipolar_y = (-right_P[2:3, :] - right_P[0:1, :] * right_epipolar_x) / right_P[1:2, :]

    colors = np.random.rand(x2.shape[0])
    plt.figure(figsize=(30, 8))
    plt.subplot(121)
    plt.imshow(img1_rgb, cmap=None if len(img1_rgb.shape)==3 else plt.get_cmap('gray'))
    plt.scatter(x1[:, 0], x1[:, 1], s=50, c=colors, edgecolors='w')
    plt.subplot(122)
    # plt.figure(figsize=(30, 8))
    plt.imshow(img2_rgb, cmap=None if len(img2_rgb.shape)==3 else plt.get_cmap('gray'))
    plt.plot(right_epipolar_x, right_epipolar_y)
    plt.scatter(x2[:, 0], x2[:, 1], s=50, c=colors, edgecolors='w')
    plt.xlim(0, im_shape[1]-1)
    plt.ylim(im_shape[0]-1, 0)
    plt.show()

def show_epipolar_rui_gtEst(x1, x2, img1_rgb, img2_rgb, F_gt, F_est, im_shape, title_append='', 
                        emphasis_idx=[], label_text=False, weights=None, if_show=True,
                        linewidth=1.0):
    N_points = x1.shape[0]
    x1_homo = utils_misc.homo_np(x1)
    x2_homo = utils_misc.homo_np(x2)
    right_P = np.matmul(F_gt, x1_homo.T)
    right_epipolar_x = np.tile(np.array([[0], [1]]), N_points) * im_shape[1]
    # Using the eqn of line: ax+by+c=0; y = (-c-ax)/b, http://ai.stanford.edu/~mitul/cs223b/draw_epipolar.m
    right_epipolar_y = (-right_P[2:3, :] - right_P[0:1, :] * right_epipolar_x) / right_P[1:2, :]

    # colors = get_spaced_colors(x2.shape[0])
    # colors = np.random.random((x2.shape[0], 3))
    # plt.figure(figsize=(60, 8))
    plt.figure(figsize=(30, 4))
    plt.imshow(img2_rgb, cmap=None if len(img2_rgb.shape)==3 else plt.get_cmap('gray'))

    plt.plot(right_epipolar_x, right_epipolar_y, 'b', linewidth=linewidth, zorder=1)
    if weights is None:
        print(f"x2: {x2.shape}")
        plt.scatter(x2[:, 0], x2[:, 1], s=50, c='r', edgecolors='w', zorder=2)
    else:
        plt.scatter(x2[:, 0], x2[:, 1], s=weights*10000, c='r', edgecolors='w', zorder=2)

    if emphasis_idx:
        for idx in emphasis_idx:
            plt.scatter(x2[idx, 0], x2[idx, 1], s=80, color='y', edgecolors='w')



    if label_text:
        for idx in range(N_points):
            plt.text(x2[idx, 0], x2[idx, 1]-10, str(idx), fontsize=20, fontweight='extra bold', color='w')

    right_P = np.matmul(F_est, x1_homo.T)
    right_epipolar_x = np.tile(np.array([[0], [1]]), N_points) * im_shape[1]
    right_epipolar_y = (-right_P[2:3, :] - right_P[0:1, :] * right_epipolar_x) / right_P[1:2, :]
    plt.plot(right_epipolar_x, right_epipolar_y, 'g', linewidth=linewidth, zorder=1) # 'r'

    plt.xlim(0, im_shape[1]-1)
    plt.ylim(im_shape[0]-1, 0)
    # plt.title('Blue lines for GT F; Red lines for est. F. -- '+title_append)
    plt.title(f"{title_append}")
    if if_show:
        plt.show()

def show_epipolar_normalized(x1, x2, img1_rgb, img2_rgb, F_gt, im_shape):
    N_points = x1.shape[0]
    x1_homo = utils_misc.homo_np(x1)
    x2_homo = utils_misc.homo_np(x2)
    right_P = np.matmul(F_gt, x1_homo.T)
    right_epipolar_x = np.tile(np.array([[-1.], [1.]]), N_points) * im_shape[1]
    # Using the eqn of line: ax+by+c=0; y = (-c-ax)/b, http://ai.stanford.edu/~mitul/cs223b/draw_epipolar.m
    right_epipolar_y = (-right_P[2:3, :] - right_P[0:1, :] * right_epipolar_x) / right_P[1:2, :]

    colors = np.random.rand(x2.shape[0])
    plt.figure(figsize=(30, 8))
    plt.subplot(121)
#     plt.imshow(img1_rgb)
#     plt.scatter(x1[:, 0]*f+W/2., x1[:, 1]*f+H/2., s=50, c=colors, edgecolors='w')
    plt.scatter(x1[:, 0], x1[:, 1], s=50, c=colors, edgecolors='w')
    plt.xlim(-im_shape[1], im_shape[1])
    plt.ylim(im_shape[0], -im_shape[0])
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.subplot(122)
#     plt.imshow(img2_rgb)
    plt.plot(right_epipolar_x, right_epipolar_y)
    plt.scatter(x2[:, 0], x2[:, 1], s=50, c=colors, edgecolors='w')
#     plt.axis('equal')
    plt.xlim(-im_shape[1], im_shape[1])
    plt.ylim(im_shape[0], -im_shape[0])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def get_spaced_colors(n):
    max_value = 16581375 #255**3
    interval = int(max_value / (n-1))
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    
    return np.stack([np.array([int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)]) for i in colors]) / 255.