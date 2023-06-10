import argparse
import time
import os

import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width


class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def get_output_filename(output_dir, base_filename):
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 检查文件名是否已经存在，如果存在则添加后缀
    output_path = os.path.join(output_dir, base_filename)
    base_name, extension = os.path.splitext(base_filename)
    counter = 2
    while os.path.exists(output_path):
        output_path = os.path.join(output_dir, base_name + '_' + str(counter) + extension)
        counter += 1

    return output_path


def run_demo(net, image_provider, height_size, cpu, track, smooth, show_fps, save, output_dir, disable_board):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1

    fps = 0
    start_time = time.time()
    frame_count = 0

    output_video = None
    video_writer = None

    for img in image_provider:
        frame_count += 1

        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        for pose in current_poses:
            pose.draw(img)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            if not disable_board:
                cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                            (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        # if show_fps:
        #     fps = frame_count / (time.time() - start_time)
        #     cv2.putText(img, 'FPS: {:.2f}'.format(fps), (8, 16), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

        if show_fps:
            # 计算帧率
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

            # 在画面上显示实时帧率
            cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if save:
            if output_video is None:
                output_filename = get_output_filename(output_dir, 'output.avi')
                output_video = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'MJPG'), 25,
                                               (img.shape[1], img.shape[0]))
            output_video.write(img)

        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)

        if cv2.waitKey(delay) == ord('q') or cv2.waitKey(delay) == 27:
            break


    if save and output_video is not None:
        output_video.release()
        print("result saved in {}".format(output_filename))

    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description='OpenPose Demo')
    parser.add_argument('--model', type=str, default='models/checkpoint_iter_370000.pth', help='model path')
    parser.add_argument('--height_size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--source', type=str, default='1', help='video source')
    parser.add_argument('--disable_track', action='store_false', help='track pose id')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--smooth', type=bool, default=True, help='smooth pose keypoints')
    parser.add_argument('--show_fps', action='store_true', help='show FPS')
    parser.add_argument('--save', type=bool, default=True, help='save output video')
    parser.add_argument('--output_dir', type=str, default='runs', help='output directory')
    parser.add_argument('--disable_board', action='store_true', help='Disable board')
    return parser.parse_args()


if __name__ == '__main__':
    parser_opt = parse_args()
    parser_opt.video = False

    if not os.path.exists(parser_opt.model):
        print(f"Model file '{parser_opt.model}' does not exist.")
        exit()

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(parser_opt.model, map_location='cpu')
    load_state(net, checkpoint)

    source = parser_opt.source
    if source.isdigit():
        source = int(source)

    if isinstance(source, int) or source.endswith('.txt'):  # video or webcam
        image_provider = VideoReader(source)
    elif os.path.isdir(source):  # image files
        image_files = []
        valid_image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        for file_name in os.listdir(source):
            extension = os.path.splitext(file_name)[1].lower()
            if extension in valid_image_extensions:
                image_files.append(os.path.join(source, file_name))
        image_provider = ImageReader(image_files)
    else:  # single image
        image_provider = ImageReader([source])

    run_demo(net, image_provider, parser_opt.height_size, cpu=parser_opt.cpu, track=parser_opt.disable_track,
             smooth=parser_opt.smooth, show_fps=parser_opt.show_fps, save=parser_opt.save,
             output_dir=parser_opt.output_dir, disable_board=parser_opt.disable_board)
