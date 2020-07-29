import numpy as np
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
from tqdm import tqdm


def get_one_frame(image_frames, frame_w, frame_h, img_w, img_h):
    # frame = np.zeros((frame_h, frame_w))
    # for i, row in enumerate(image_frames):

        # for j, col in enumerate(row):
    img_list = []
    for i, en in enumerate(image_frames):
        img = _read_image(en, sizer_w=img_w, sizer_h=img_h)
        img_list.append(img)
    frame = np.stack(img_list, axis=0) # [4, h, w]
    print(f"frame: {frame.shape}")
    # frame = frame.transpose([1,2,0,3])
    # frame = frame.reshape((1, img_h, -1, 3))
    # frame = frame.reshape((frame_h, frame_w, 3))
    print(f"frame: {frame.shape}")
        # frame[0:frame_h//2, 0:frame_w//2] = img
    return frame

def get_frames(frames_list, frame_w, frame_h, img_w, img_h, iter=4):
    """
    frames_list: [N x 4]
    """
    frames = []
    # frame_black = np.random.randint(0, 256, 
    #                         (frame_h, frame_w, 3), 
    #                         dtype=np.uint8)
    frame_blocks = np.zeros((frame_h, frame_w,3), dtype=np.uint8)
    for i in range(iter):
        img_list = []
        files = frames_list[:,i]
        for j, en in tqdm(enumerate(files)):
            # print(f"file: {en}")
            img = _read_image(en, sizer_w=img_w, sizer_h=img_h)
            img_list.append(img)
        frame = np.stack(img_list, axis=0) # [4, h, w]
        # print(f"frame: {frame.shape}")
        # frame = frame.reshape((img_h, img_w, 2, 2, 3))
        # # frame = frame.transpose([2,0,3,1,4])
        # frame = frame.transpose([0,2,1,3,4])
        # frame = frame.reshape((img_h, 2, frame_w, 3))
        # frame = frame.reshape((frame_h, frame_w, 3))
        # frame = frame.astype(np.uint8).copy()
        frame_blocks[0:img_h,0:img_w] = frame[0]
        frame_blocks[0:img_h,img_w:frame_w] = frame[1]
        frame_blocks[img_h:frame_h,0:img_w] = frame[2]
        frame_blocks[img_h:frame_h,img_w:frame_w] = frame[3]
        # print(f"frame: {frame}")
        # frame_black = frame.copy()
        # print(f"frame: {frame.shape}")
        # frames.append(frame)
        frames.append(frame_blocks)
        # frame[0:frame_h//2, 0:frame_w//2] = img
    return frames


def output_video(frames, width, height, FPS=24, filename="output.mp4"):
    fourcc = VideoWriter_fourcc(*'MP4V')
    video = VideoWriter(filename, fourcc, float(FPS), (width, height))

    for i, f in tqdm(enumerate(frames)):
        # frame = np.random.randint(0, 256, 
        #                         (height, width, 3), 
        #                         dtype=np.uint8)
        # print(f"frame: {frame.shape}")
        video.write(f)
    video.release()
    pass


def _read_image(path, sizer_w=0, sizer_h=0):
    input_image = cv2.imread(path)
    if sizer_h > 0 and sizer_w > 0:
        input_image = cv2.resize(input_image, (sizer_w, sizer_h),
                                    interpolation=cv2.INTER_AREA)
    H, W = input_image.shape[0], input_image.shape[1]
    # input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # input_image = input_image.astype('float32') / 255.0
    return input_image

def form_video_frame(image_frames, frame_w, frame_h):
    pass

if __name__ == "__main__":

    width = 1241//2
    height = 376//2
    frame_w = width*2
    frame_h = height*2
    FPS = 24
    seconds = 10
    n_frames = 1000
    video_name = 'plots/kitti_cp_bases_1k_0228.mp4'

    BASE_PATH = "/home/yyjau/Documents/deepSfm/"
    ours_plot = f"{BASE_PATH}/plots/vis_paper/Sp-Df-fp-k/mask_conf_Sp-Df-fp-k_all_fr_0.png"
    ours_end_plot = f"{BASE_PATH}/plots/vis_paper/Sp-Df-fp-end-k/mask_conf_Sp-Df-fp-end-k_all_fr_0.png"
    si_base_plot = f"{BASE_PATH}/plots/vis_paper/Si-Df-fp-k/mask_conf_Si-Df-fp-k_all_fr_0.png"
    kitti_img = f"/data/kitti/kitti_dump/odo_corr_dump_siftIdx_npy_delta1235810_full_qualityRE2/09_02/000000.jpg"

    # image_frames = [[kitti_img, si_base_plot],[ours_plot, ours_end_plot]]
    image_frames = [kitti_img, si_base_plot, ours_plot, ours_end_plot]
    frames_base_path = [kitti_img[:-10], si_base_plot[:-5], ours_plot[:-5], ours_end_plot[:-5]]
    frames_list = []
    frames_list.append([f"{frames_base_path[0]}{i:06}.jpg" for i in range(n_frames)])
    for j in range(1,4):
        frames_list.append([f"{frames_base_path[j]}{i}.png" for i in range(n_frames)])
    frames_list = np.array(frames_list, dtype=np.str)
    print(f"frames_list: {len(frames_list)}")
    # frames_list = [[f"{p for  in range(n_frames)], si_base_plot, ours_plot, ours_end_plot]
    frames = get_frames(frames_list, frame_w=frame_w, frame_h=frame_h, img_w=width, img_h=height, iter=n_frames)
    print(f"frames_list: {len(frames)}")
    output_video(frames, width=frame_w, height=frame_h, FPS=24, filename=video_name )


    # img = cv2.imread(ours_plot)
    # print(img)
    # img = img.astype('uint8')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = _read_image(ours_end_plot, sizer_w=width, sizer_h=height)
    # print(f"image: {img.shape}")
    # frames = get_one_frame(image_frames, frame_w=width*2, frame_h=height*2, img_w=width, img_h=height)
    # output_video(frames)


    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # voObj = cv2.VideoWriter('output.mp4',fourcc, 15.0, (1280,360))
    # fourcc = cv2.VideoWriter_fourcc(*'H264')