import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.common import as_intrinsics_matrix, normalization


def get_dataset(cfg, args, scale, device="cuda:0"):
    """
    Wrapper that retrieves dataset class based on configuration.

    """
    return dataset_dict[cfg["dataset"]](cfg, args, scale, device=device)


def get_normal_map(depth_data, K, kernel_size=9, color_std=100, spatial_std=9):
    """
    Applies bilateral filter and cross product for normal information.

    """

    H, W = depth_data.shape

    # smooth depth while preserving edges
    depth_filtered = cv2.bilateralFilter(depth_data, kernel_size, color_std, spatial_std)

    # store vector differences
    f = np.zeros((3, H, W))
    t = np.zeros((3, H, W))

    # get pixel position vector
    x, y = np.meshgrid(np.arange(0, W), np.arange(0, H))
    x = x.reshape([-1])
    y = y.reshape([-1])
    xyz = np.vstack((x, y, np.ones_like(x)))

    # project from 2D to 3D
    pts_3d = np.dot(np.linalg.inv(K), xyz * depth_filtered.reshape([-1]))
    pts_3d_world = pts_3d.reshape((3, H, W))

    # get vector differences
    f[:, 1:-1, 1:-1] = pts_3d_world[:, 1 : H - 1, 2:W] - pts_3d_world[:, 1 : H - 1, 1 : W - 1]
    t[:, 1:-1, 1:-1] = pts_3d_world[:, 2:H, 1 : W - 1] - pts_3d_world[:, 1 : H - 1, 1 : W - 1]

    # get normal and normalize to unit vector
    normal_data = np.cross(f, t, axisa=0, axisb=0)
    normal_data = normalization(normal_data)

    return normal_data.astype(np.float32)


class BaseDataset(Dataset):
    def __init__(self, cfg, args, scale, device="cuda:0"):
        super(BaseDataset, self).__init__()
        self.name = cfg["dataset"]
        self.num_sensors = cfg["num_sensors"]
        self.device = device
        self.scale = scale
        self.png_depth_scale = cfg["cam"]["png_depth_scale"]

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = (
            cfg["cam"]["H"],
            cfg["cam"]["W"],
            cfg["cam"]["fx"],
            cfg["cam"]["fy"],
            cfg["cam"]["cx"],
            cfg["cam"]["cy"],
        )

        # resize the input images to crop_size (variable name used in lietorch)
        if "crop_size" in cfg["cam"]:
            crop_size = cfg["cam"]["crop_size"]
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx * self.fx
            self.fy = sy * self.fy
            self.cx = sx * self.cx
            self.cy = sy * self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]

        # croping will change H, W, cx, cy, so need to change here
        if cfg["cam"]["crop_edge"] > 0:
            self.H -= cfg["cam"]["crop_edge"] * 2
            self.W -= cfg["cam"]["crop_edge"] * 2
            self.cx -= cfg["cam"]["crop_edge"]
            self.cy -= cfg["cam"]["crop_edge"]

        self.distortion = np.array(cfg["cam"]["distortion"]) if "distortion" in cfg["cam"] else None
        self.crop_size = cfg["cam"]["crop_size"] if "crop_size" in cfg["cam"] else None  # TODO: maybe remove

        if args.input_folder is None:
            self.input_folder = cfg["data"]["input_folder"]
        else:
            self.input_folder = args.input_folder

        self.crop_edge = cfg["cam"]["crop_edge"]

    def __len__(self):
        return self.n_img

    def __getitem__(self, index):
        ret = {}
        prefix_keys = ["", "2_", "3_"]

        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        kern = 5
        kx, ky = cv2.getDerivKernels(0, 1, kern)
        norm_factor = (kx @ ky.T)[:, kern // 2 + 1 :].sum()

        edge = self.crop_edge

        # depth data
        depth_path = self.depth_paths[0][index]
        if ".png" in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if self.name == "SevenScenes":
            depth_data[depth_data == 65535] = 0

        depth_data = depth_data.astype(np.float32) / self.png_depth_scale
        depth_data = depth_data * self.scale

        H, W = depth_data.shape

        # color data
        color_path = self.color_paths[index]
        color_data = cv2.imread(color_path)
        if self.distortion is not None:
            # undistortion is only applied on color image, not depth!
            color_data = cv2.undistort(color_data, K, self.distortion)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.0
        color_data = cv2.resize(color_data, (W, H))

        # GT depth data
        if self.gt_depth_paths:
            gt_depth_path = self.gt_depth_paths[index]
            if ".png" in gt_depth_path:
                gt_depth_data = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED)
                gt_depth_data = gt_depth_data.astype(np.float32) / self.png_depth_scale
        else:
            gt_depth_data = depth_data

        # local ray data
        local_data = np.zeros((self.H, self.W, 3), np.float32)
        x, y = np.meshgrid(np.arange(0, self.W), np.arange(0, self.H))
        local_data[:, :, 0] = (x - K[0, 2]) / K[0, 0]
        local_data[:, :, 1] = (y - K[1, 2]) / K[1, 1]
        local_data[:, :, 2] = 1
        local_data = normalization(local_data)

        color_data = torch.from_numpy(color_data)
        gt_depth_data = torch.from_numpy(gt_depth_data)
        local_data = torch.from_numpy(local_data)

        # follow the pre-processing step in lietorch, actually is resize
        if self.crop_size is not None:
            color_data = color_data.permute(2, 0, 1)
            gt_depth_data = F.interpolate(gt_depth_data[None, None], self.crop_size, mode="nearest")[0, 0]
            color_data = F.interpolate(color_data[None], self.crop_size, mode="bilinear", align_corners=True)[0]
            color_data = color_data.permute(1, 2, 0).contiguous()

        if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            gt_depth_data = gt_depth_data[edge:-edge, edge:-edge]

        for i in range(self.num_sensors):
            # depth data
            depth_path = self.depth_paths[i][index]
            if ".png" in depth_path:
                depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth_data = depth_data.astype(np.float32) / self.png_depth_scale
            depth_data = depth_data * self.scale

            depth_data = torch.from_numpy(depth_data)

            # follow the pre-processing step in lietorch, actually is resize
            if self.crop_size is not None:
                depth_data = F.interpolate(depth_data[None, None], self.crop_size, mode="nearest")[0, 0]

            # crop image edge, there are invalid value on the edge of the color image
            if edge > 0:
                depth_data = depth_data[edge:-edge, edge:-edge]

            # error data
            error_data = torch.abs(gt_depth_data - depth_data)
            error_data[depth_data == 0] = 0

            # normal data
            normal_data = get_normal_map(depth_data.numpy(), K)

            # angle data
            angle_data = np.sum(local_data.numpy() * normal_data, axis=2)

            # gradient data
            dx_data = cv2.Sobel(depth_data.numpy(), cv2.CV_32F, 1, 0, ksize=kern) / norm_factor
            dy_data = cv2.Sobel(depth_data.numpy(), cv2.CV_32F, 0, 1, ksize=kern) / norm_factor

            normal_data = torch.from_numpy(normal_data)
            angle_data = torch.from_numpy(angle_data)
            dx_data = torch.from_numpy(dx_data)
            dy_data = torch.from_numpy(dy_data)

            ret.update(
                {
                    f"{prefix_keys[i]}depth": depth_data.to(self.device),
                    f"{prefix_keys[i]}normal": normal_data.to(self.device),
                    f"{prefix_keys[i]}error": error_data.to(self.device),
                    f"{prefix_keys[i]}angle": angle_data.to(self.device),
                    f"{prefix_keys[i]}dx": dx_data.to(self.device),
                    f"{prefix_keys[i]}dy": dy_data.to(self.device),
                }
            )

        pose = self.poses[index]
        pose[:3, 3] *= self.scale

        ret.update(
            {
                "idx": index,
                "gt_c2w": pose.to(self.device),
                "color": color_data.to(self.device),
                "normal": normal_data.to(self.device),
                "local": local_data.to(self.device),
                "gt_depth": gt_depth_data.to(self.device),
            }
        )

        return ret


class Replica_SenFuNet(BaseDataset):
    """
    Replica dataset as rendered in Sensor Fusion. Uses left camera view.

    """

    def __init__(self, cfg, args, scale, device="cuda:0"):
        super(Replica_SenFuNet, self).__init__(cfg, args, scale, device)

        self.depth_paths = []  # left_sgm_depth, left_depth_noise_1.0, left_psmnet_depth, left_routing_confidence
        self.depth_paths.append(
            sorted(
                glob.glob(f"{self.input_folder}/left_depth_noise_1.0/*.png"),
                key=lambda s: int(s.rsplit("/")[-1][:-4]),
            )
        )
        self.depth_paths.append(
            sorted(
                glob.glob(f"{self.input_folder}/left_psmnet_depth/*.png"),
                key=lambda s: int(s.rsplit("/")[-1][:-4]),
            )
        )

        self.color_paths = sorted(
            glob.glob(f"{self.input_folder}/left_rgb/*.png"),
            key=lambda s: int(s.rsplit("/")[-1][:-4]),
        )

        self.gt_depth_paths = sorted(
            glob.glob(f"{self.input_folder}/left_depth_gt/*.png"),
            key=lambda s: int(s.rsplit("/")[-1][:-4]),
        )

        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}/left_camera_matrix")

    def load_poses(self, path):
        pose_files = sorted(glob.glob(f"{path}/*.txt"), key=lambda s: int(s.rsplit("/")[-1][:-4]))

        self.poses = []

        H_yz = np.eye(4)
        H_yz[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]  # rotation around x axis

        for pose_file in pose_files:
            with open(pose_file, "rb") as infile:
                index = 0
                h_mat_line = b""
                for line in infile:
                    if b"#" not in line:
                        h_mat_line += line[:-1] + b" "
                        index += 1
                    if index == 4:
                        c2w = np.array(h_mat_line.split(), np.float64).reshape(4, 4)
                        c2w = H_yz @ c2w
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class TUM_RGBD(BaseDataset):
    def __init__(self, cfg, args, scale, device="cuda:0"):
        super(TUM_RGBD, self).__init__(cfg, args, scale, device)

        self.color_paths, self.depth_paths, self.poses = self.loadtum(self.input_folder, frame_rate=32)

        self.gt_depth_paths = None

        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        """Read list data."""
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """Pair images, depths, and poses."""
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """Read video data in tum-rgbd format."""
        if os.path.isfile(os.path.join(datapath, "groundtruth.txt")):
            pose_list = os.path.join(datapath, "groundtruth.txt")
        elif os.path.isfile(os.path.join(datapath, "pose.txt")):
            pose_list = os.path.join(datapath, "pose.txt")

        image_list = os.path.join(datapath, "rgb.txt")
        depth_list = os.path.join(datapath, "depth.txt")

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths = [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose @ c2w
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses += [c2w]

        return images, [depths], poses

    def pose_matrix_from_quaternion(self, pvec):
        """Convert 4x4 pose matrix to (t, q)."""
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose


class SevenScenes(BaseDataset):
    def __init__(self, cfg, args, scale, device="cuda:0"):
        super(SevenScenes, self).__init__(cfg, args, scale, device)

        self.color_paths, self.depth_paths, self.poses = self.load7scenes(self.input_folder)

        self.gt_depth_paths = None

        self.n_img = len(self.color_paths)

    def load7scenes(self, data_path):
        """Read video data in 7-Scenes format."""

        data_path = Path(data_path)

        image_list = sorted([str(data_file) for data_file in data_path.glob("*.color.png")])
        depth_list = sorted([str(data_file) for data_file in data_path.glob("*.depth.png")])
        pose_list = sorted([str(data_file) for data_file in data_path.glob("*.pose.txt")])

        poses = []

        for pose_file in pose_list:
            c2w = np.loadtxt(pose_file)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses += [c2w]

        return image_list, [depth_list], poses


dataset_dict = {
    "replica_senfunet": Replica_SenFuNet,
    "tumrgbd": TUM_RGBD,
    "7scenes": SevenScenes,
}
