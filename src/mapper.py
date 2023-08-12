import os
import time

import cv2
import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from src.common import (
    get_camera_from_tensor,
    get_samples,
    get_tensor_from_camera,
    random_select,
)
from src.utils.datasets import get_dataset
from src.utils.visualizer import Visualizer


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class Mapper(object):
    """Mapper thread."""

    def __init__(self, cfg, args, slam):
        setup_seed(142857)

        self.cfg = cfg
        self.args = args
        self.prefix_keys = ["", "2_", "3_"]

        # control the features used in mapping process TODO: initialize in uncle_slam.py
        self.feature_maps = ["color", "local"]
        depth_features = ["depth", "normal", "error", "angle", "dx", "dy"]
        for i in range(self.cfg["num_sensors"]):
            prefix = self.prefix_keys[i]
            self.feature_maps += [f"{prefix}{feature}" for feature in depth_features]

        self.idx = slam.idx
        self.c = slam.shared_c
        self.bound = slam.bound
        self.logger = slam.logger
        self.mesher = slam.mesher
        self.output = slam.output
        self.verbose = slam.verbose
        self.renderer = slam.renderer
        self.low_gpu_mem = slam.low_gpu_mem
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list
        self.mapping_first_frame = slam.mapping_first_frame

        self.scale = cfg["scale"]
        self.occupancy = cfg["occupancy"]
        self.sync_method = cfg["sync_method"]

        self.device = cfg["mapping"]["device"]
        self.fix_fine = cfg["mapping"]["fix_fine"]
        self.eval_rec = cfg["meshing"]["eval_rec"]
        self.BA = False  # Even if BA is enabled, it starts only when there are at least 4 keyframes
        self.BA_cam_lr = cfg["mapping"]["BA_cam_lr"]
        self.mesh_freq = cfg["mapping"]["mesh_freq"]
        self.ckpt_freq = cfg["mapping"]["ckpt_freq"]
        self.fix_color = cfg["mapping"]["fix_color"]
        self.mapping_pixels = cfg["mapping"]["pixels"]
        self.num_joint_iters = cfg["mapping"]["iters"]
        self.clean_mesh = cfg["meshing"]["clean_mesh"]
        self.every_frame = cfg["mapping"]["every_frame"]
        self.color_refine = cfg["mapping"]["color_refine"]
        self.w_color_loss = cfg["mapping"]["w_color_loss"]
        self.keyframe_every = cfg["mapping"]["keyframe_every"]
        self.fine_iter_ratio = cfg["mapping"]["fine_iter_ratio"]
        self.middle_iter_ratio = cfg["mapping"]["middle_iter_ratio"]
        self.use_color_in_mapping = cfg["mapping"]["use_color_in_mapping"]
        self.mapping_window_size = cfg["mapping"]["mapping_window_size"]
        self.no_vis_on_first_frame = cfg["mapping"]["no_vis_on_first_frame"]
        self.no_log_on_first_frame = cfg["mapping"]["no_log_on_first_frame"]
        self.no_mesh_on_first_frame = cfg["mapping"]["no_mesh_on_first_frame"]
        self.frustum_feature_selection = cfg["mapping"]["frustum_feature_selection"]
        self.keyframe_selection_method = cfg["mapping"]["keyframe_selection_method"]
        self.save_selected_keyframes_info = cfg["mapping"]["save_selected_keyframes_info"]
        if self.save_selected_keyframes_info:
            self.selected_keyframes = {}

        self.keyframe_dict = []
        self.keyframe_list = []
        self.frame_reader = get_dataset(cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        if "Demo" not in self.output:  # disable this visualization in demo
            self.visualizer = Visualizer(
                freq=cfg["mapping"]["vis_freq"],
                inside_freq=cfg["mapping"]["vis_inside_freq"],
                vis_dir=os.path.join(self.output, "mapping_vis"),
                renderer=self.renderer,
                verbose=self.verbose,
                device=self.device,
            )
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = (
            slam.H,
            slam.W,
            slam.fx,
            slam.fy,
            slam.cx,
            slam.cy,
        )

    def get_mask_from_c2w(self, c2w, key, val_shape, depth_np):
        """
        Frustum feature selection based on current camera pose and depth image.

        Args:
            c2w (tensor): camera pose of current frame.
            key (str): name of this feature grid.
            val_shape (tensor): shape of the grid.
            depth_np (List[numpy.array]): depth maps of current frame(s).

        Returns:
            mask (tensor): mask for selected optimizable feature.
            points (tensor): corresponding point coordinates.

        """
        (
            H,
            W,
            fx,
            fy,
            cx,
            cy,
        ) = (
            self.H,
            self.W,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
        )
        X, Y, Z = torch.meshgrid(
            torch.linspace(self.bound[0][0], self.bound[0][1], val_shape[2]),
            torch.linspace(self.bound[1][0], self.bound[1][1], val_shape[1]),
            torch.linspace(self.bound[2][0], self.bound[2][1], val_shape[0]),
        )

        points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
        points_bak = points.clone()
        c2w = c2w.cpu().numpy()
        w2c = np.linalg.inv(c2w)
        ones = np.ones_like(points[:, 0]).reshape(-1, 1)
        homo_vertices = np.concatenate([points, ones], axis=1).reshape(-1, 4, 1)
        cam_cord_homo = w2c @ homo_vertices
        cam_cord = cam_cord_homo[:, :3]
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]).reshape(3, 3)
        cam_cord[:, 0] *= -1
        uv = K @ cam_cord
        z = uv[:, -1:] + 1e-5
        uv = uv[:, :2] / z
        uv = uv.astype(np.float32)

        edge = 0
        mask = (uv[:, 0] < W - edge) * (uv[:, 0] > edge) * (uv[:, 1] < H - edge) * (uv[:, 1] > edge)

        depths_list = []

        # get depths for each ray for each depth map
        for depth in depth_np:
            remap_chunk = int(3e4)
            depths = []
            for i in range(0, uv.shape[0], remap_chunk):
                depths += [
                    cv2.remap(
                        depth,
                        uv[i : i + remap_chunk, 0],
                        uv[i : i + remap_chunk, 1],
                        interpolation=cv2.INTER_LINEAR,
                    )[:, 0].reshape(-1, 1)
                ]
            depths_list.append(np.concatenate(depths, axis=0))

        # For ray with depth==0, fill it with maximum depth
        for depths in depths_list:
            zero_mask = depths == 0
            depths[zero_mask] = np.max(depths)

        # depth test
        mask = mask & (0 <= -z[:, :, 0])
        mask_depth = np.zeros_like(mask)
        for depths in depths_list:
            mask_depth = mask_depth | (-z[:, :, 0] <= depths + 0.5)
        mask = mask & mask_depth
        mask = mask.reshape(-1)

        # add feature grid near cam center
        ray_o = c2w[:3, 3]
        ray_o = torch.from_numpy(ray_o).unsqueeze(0)

        dist = points_bak - ray_o
        dist = torch.sum(dist * dist, axis=1)
        mask2 = dist < 0.5 * 0.5
        mask2 = mask2.cpu().numpy()
        mask = mask | mask2

        points = points[mask]
        mask = mask.reshape(val_shape[2], val_shape[1], val_shape[0])
        return mask

    def keyframe_selection_overlap(self, frame_dict, keyframe_dict, k, N_samples=16, pixels=100):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            frame_dict: dictionary of images / poses for the current frame.
                est_c2w (tensor): estimated pose / camera tensor.
                depth (tensor): measured depth image of the current frame. (H,W)
                ...
            k (int): number of overlapping keyframes to select.
            N_samples (int, optional): number of samples/points per ray. Defaults to 16.
            pixels (int, optional): number of pixels to sparsely sample
                from the image of the current camera. Defaults to 100.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.

        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        rays_o, rays_d, ray_dict = get_samples(
            0,
            H,
            0,
            W,
            pixels,
            H,
            W,
            fx,
            fy,
            cx,
            cy,
            frame_dict,
            self.feature_maps,
            self.device,
        )

        # get first sensor depth
        gt_depth = ray_dict["depth"].reshape(-1, 1)
        corr_factor = torch.zeros_like(gt_depth)
        corr_factor += gt_depth != 0

        # add additional depth sensors if available
        for i in range(1, self.cfg["num_sensors"]):
            prefix = self.prefix_keys[i]
            sensor_depth = ray_dict[f"{prefix}depth"].reshape(-1, 1)
            corr_factor += sensor_depth != 0
            gt_depth += sensor_depth

        # average depth readings for each pixel depending on available data
        gt_depth[corr_factor != 0] /= corr_factor[corr_factor != 0]

        # get points along ray
        gt_depth = gt_depth.repeat(1, N_samples)
        t_vals = torch.linspace(0.0, 1.0, steps=N_samples).to(device)
        near = gt_depth * 0.8
        far = gt_depth + 0.5
        z_vals = near * (1.0 - t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
        vertices = pts.reshape(-1, 3).cpu().numpy()

        list_keyframe = []

        # calculate number of projected points in other keyframes
        for keyframeid, keyframe in enumerate(keyframe_dict):
            c2w = keyframe["est_c2w"].cpu().numpy()
            w2c = np.linalg.inv(c2w)
            ones = np.ones_like(vertices[:, 0]).reshape(-1, 1)
            homo_vertices = np.concatenate([vertices, ones], axis=1).reshape(-1, 4, 1)  # (N, 4)
            cam_cord_homo = w2c @ homo_vertices  # (N, 4, 1)=(4,4)*(N, 4, 1)
            cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]).reshape(3, 3)
            cam_cord[:, 0] *= -1
            uv = K @ cam_cord
            z = uv[:, -1:] + 1e-5
            uv = uv[:, :2] / z
            uv = uv.astype(np.float32)
            edge = 20
            mask = (uv[:, 0] < W - edge) * (uv[:, 0] > edge) * (uv[:, 1] < H - edge) * (uv[:, 1] > edge)
            mask = mask & (z[:, :, 0] < 0)
            mask = mask.reshape(-1)
            percent_inside = mask.sum() / uv.shape[0]
            list_keyframe.append({"id": keyframeid, "percent_inside": percent_inside})

        list_keyframe = sorted(list_keyframe, key=lambda i: i["percent_inside"], reverse=True)
        selected_keyframe_list = [dic["id"] for dic in list_keyframe if dic["percent_inside"] > 0.00]
        selected_keyframe_list = list(np.random.permutation(np.array(selected_keyframe_list))[:k])
        return selected_keyframe_list

    def optimize_map(self, num_joint_iters, lr_factor, idx, frame_dict, keyframe_dict, keyframe_list):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation and camera poses(if local BA enabled).

        Args:
            num_joint_iters (int): number of mapping iterations.
            lr_factor (float): the factor to times on current lr.
            idx (int): the index of current frame
            frame_dict (Dict): dictionary of images / poses for the current frame.
                gt_c2w (tensor): ground truth pose.
                est_c2w (tensor): estimated camera to world matrix of current frame.
                color (tensor): measured color image of the current frame. (H,W,3)
                Per sensor:
                    depth (tensor): measured depth image of the current frame. (H,W)
                    normal (tensor): smoothed normal map from depth image. (H,W,3)
                    local (tensor): local ray direction in camera reference. (H,W,3)
                    error (tensor): absolute difference GT vs. measured depth (vis only). (H, W)
                    angle (tensor): angle difference between normal and local ray. (H,W)
            keyframe_dict (list): list of keyframes info dictionary.
            keyframe_list (list): list ofkeyframe index.

        Returns:
            cur_c2w/None (tensor/None): return the updated cur_c2w, return None if no BA
        """

        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        c = self.c
        cfg = self.cfg
        device = self.device
        bottom = torch.from_numpy(np.array([0, 0, 0, 1.0]).reshape([1, 4])).type(torch.float32).to(device)

        # get list of frames to jointly optimize
        if len(keyframe_dict) == 0:
            optimize_frame = []
        else:
            if self.keyframe_selection_method == "global":
                num = self.mapping_window_size - 2
                optimize_frame = random_select(len(self.keyframe_dict) - 1, num)
            elif self.keyframe_selection_method == "overlap":
                num = self.mapping_window_size - 2
                optimize_frame = self.keyframe_selection_overlap(frame_dict, keyframe_dict[:-1], num)

        # add the last keyframe and the current frame(use -1 to denote)
        oldest_frame = None
        if len(keyframe_list) > 0:
            optimize_frame = optimize_frame + [len(keyframe_list) - 1]
            oldest_frame = min(optimize_frame)
        optimize_frame += [-1]

        # save which keyframes were used at each index
        if self.save_selected_keyframes_info:
            keyframes_info = []
            for id, frame in enumerate(optimize_frame):
                if frame != -1:
                    frame_idx = keyframe_list[frame]
                    tmp_gt_c2w = keyframe_dict[frame]["gt_c2w"]
                    tmp_est_c2w = keyframe_dict[frame]["est_c2w"]
                else:
                    frame_idx = idx
                    tmp_gt_c2w = frame_dict["gt_c2w"]
                    tmp_est_c2w = frame_dict["est_c2w"]
                keyframes_info.append({"idx": frame_idx, "gt_c2w": tmp_gt_c2w, "est_c2w": tmp_est_c2w})
            self.selected_keyframes[idx] = keyframes_info

        # number of pixels to sample per frame in optimization list
        pixs_per_image = self.mapping_pixels // len(optimize_frame)

        decoders_params_list = []
        middle_grid_params = []
        fine_grid_params = []
        color_grid_params = []
        uncertainty_params = []

        gt_depth_np = []

        for i in range(cfg["num_sensors"]):
            prefix = self.prefix_keys[i]
            gt_depth_np.append(frame_dict[f"{prefix}depth"].cpu().numpy())

        # set-up uncle-slam optimization
        if self.frustum_feature_selection:
            masked_c_grad = {}
            mask_c2w = frame_dict["est_c2w"]
        for key, val in c.items():
            if not self.frustum_feature_selection:
                val = Variable(val.to(device), requires_grad=True)
                c[key] = val
                if key == "grid_middle":
                    middle_grid_params.append(val)
                elif key == "grid_fine":
                    fine_grid_params.append(val)
                elif key == "grid_color":
                    color_grid_params.append(val)
                elif key == "grid_var":
                    uncertainty_params.append(val)
            else:
                mask = self.get_mask_from_c2w(mask_c2w, key, val.shape[2:], gt_depth_np)
                mask = (
                    torch.from_numpy(mask).permute(2, 1, 0).unsqueeze(0).unsqueeze(0).repeat(1, val.shape[1], 1, 1, 1)
                )
                val = val.to(device)
                # val_grad is the optimizable part, other parameters will be fixed
                val_grad = val[mask].clone()
                val_grad = Variable(val_grad.to(device), requires_grad=True)
                masked_c_grad[key] = val_grad
                masked_c_grad[key + "mask"] = mask
                if key == "grid_middle":
                    middle_grid_params.append(val_grad)
                elif key == "grid_fine":
                    fine_grid_params.append(val_grad)
                elif key == "grid_color":
                    color_grid_params.append(val_grad)

        # whether fine decoder and color decoder is fixed
        if not self.fix_fine:
            decoders_params_list += list(self.decoders.fine_decoder.parameters())
        if not self.fix_color:
            decoders_params_list += list(self.decoders.color_decoder.parameters())
        for decoder in self.decoders.conf_decoders:
            uncertainty_params += list(decoder.parameters())
        uncertainty_params += list(self.decoders.color_conf_decoder.parameters())

        # get list of optimizable camera tensors
        if self.BA:
            camera_tensor_list = []
            gt_camera_tensor_list = []
            for frame in optimize_frame:
                # the oldest frame should be fixed to avoid drifting
                if frame != oldest_frame:
                    if frame != -1:
                        c2w = keyframe_dict[frame]["est_c2w"]
                        gt_c2w = keyframe_dict[frame]["gt_c2w"]
                    else:
                        c2w = frame_dict["est_c2w"]
                        gt_c2w = frame_dict["gt_c2w"]
                    camera_tensor = get_tensor_from_camera(c2w)
                    camera_tensor = Variable(camera_tensor.to(device), requires_grad=True)
                    camera_tensor_list.append(camera_tensor)
                    gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                    gt_camera_tensor_list.append(gt_camera_tensor)

        # set-up optimizer

        optimizer_config = [
            {"params": decoders_params_list, "lr": 0},
            {"params": middle_grid_params, "lr": 0},
            {"params": fine_grid_params, "lr": 0},
            {"params": color_grid_params, "lr": 0},
            {"params": uncertainty_params, "lr": 0},
        ]

        if self.BA:
            optimizer_config.append(
                {"params": camera_tensor_list, "lr": 0},
            )
        # The corresponding lr will be set according to which stage the optimization is in
        optimizer = torch.optim.Adam(optimizer_config)

        # run specified number of refinement iterations
        for joint_iter in range(num_joint_iters):
            if self.frustum_feature_selection:
                for key, val in c.items():
                    val_grad = masked_c_grad[key]
                    mask = masked_c_grad[key + "mask"]
                    val = val.to(device)
                    val[mask] = val_grad
                    c[key] = val

            if joint_iter <= int(num_joint_iters * self.middle_iter_ratio):
                self.stage = "middle"
            elif joint_iter <= int(num_joint_iters * self.fine_iter_ratio):
                self.stage = "fine"
            else:
                self.stage = "color"

            optimizer.param_groups[0]["lr"] = cfg["mapping"]["stage"][self.stage]["decoders_lr"] * lr_factor
            optimizer.param_groups[1]["lr"] = cfg["mapping"]["stage"][self.stage]["middle_lr"] * lr_factor
            optimizer.param_groups[2]["lr"] = cfg["mapping"]["stage"][self.stage]["fine_lr"] * lr_factor
            optimizer.param_groups[3]["lr"] = cfg["mapping"]["stage"][self.stage]["color_lr"] * lr_factor
            optimizer.param_groups[4]["lr"] = cfg["mapping"]["stage"][self.stage]["var_lr"] * lr_factor
            if self.BA:
                if self.stage == "color":
                    optimizer.param_groups[5]["lr"] = self.BA_cam_lr

            if (not (idx == 0 and self.no_vis_on_first_frame)) and ("Demo" not in self.output):
                self.visualizer.vis(
                    idx,
                    joint_iter,
                    frame_dict,
                    self.feature_maps,
                    self.c,
                    self.decoders,
                )

            optimizer.zero_grad()

            batch_dict_lists = {}

            for feature in self.feature_maps + ["rays_o", "rays_d"]:
                batch_dict_lists[feature] = []
                if feature not in ["rays_o", "rays_d"]:
                    for i in range(cfg["patch_size"] ** 2):
                        batch_dict_lists[f"{feature}_{i}"] = []

            optim_frame_dict = {}

            camera_tensor_id = 0

            # get individual rays from each frame in the bundle adjustment list
            for frame in optimize_frame:
                if frame != -1:
                    for feature in self.feature_maps:
                        optim_frame_dict[feature] = keyframe_dict[frame][feature].to(device)
                    if self.BA and frame != oldest_frame:
                        camera_tensor = camera_tensor_list[camera_tensor_id]
                        camera_tensor_id += 1
                        optim_frame_dict["est_c2w"] = get_camera_from_tensor(camera_tensor)
                    else:
                        optim_frame_dict["est_c2w"] = keyframe_dict[frame]["est_c2w"]
                else:
                    for feature in self.feature_maps:
                        optim_frame_dict[feature] = frame_dict[feature].to(device)
                    if self.BA:
                        camera_tensor = camera_tensor_list[camera_tensor_id]
                        optim_frame_dict["est_c2w"] = get_camera_from_tensor(camera_tensor)
                    else:
                        optim_frame_dict["est_c2w"] = frame_dict["est_c2w"]

                batch_rays_o, batch_rays_d, batch_dict = get_samples(
                    0,
                    H,
                    0,
                    W,
                    pixs_per_image,
                    H,
                    W,
                    fx,
                    fy,
                    cx,
                    cy,
                    optim_frame_dict,
                    self.feature_maps,
                    self.device,
                    kernel=cfg["patch_size"],
                )
                batch_dict_lists["rays_o"].append(batch_rays_o.float())
                batch_dict_lists["rays_d"].append(batch_rays_d.float())
                for feature in self.feature_maps:
                    batch_dict_lists[feature].append(batch_dict[feature].float())
                    for mlp_index in range(cfg["patch_size"] ** 2):
                        batch_dict_lists[f"{feature}_{mlp_index}"].append(batch_dict[f"{feature}_{mlp_index}"].float())

            batch_dict = {}

            batch_dict["rays_o"] = torch.cat(batch_dict_lists["rays_o"])
            batch_dict["rays_d"] = torch.cat(batch_dict_lists["rays_d"])

            for feature in self.feature_maps:
                batch_dict[feature] = torch.cat(batch_dict_lists[feature])
                for mlp_index in range(cfg["patch_size"] ** 2):
                    batch_dict[f"{feature}_{mlp_index}"] = torch.cat(batch_dict_lists[f"{feature}_{mlp_index}"])

            # should pre-filter those out of bounding box depth value
            with torch.no_grad():
                det_rays_o = batch_dict["rays_o"].clone().detach().unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = batch_dict["rays_d"].clone().detach().unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0).to(device) - det_rays_o) / det_rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = t >= batch_dict["depth"]
            for feature in self.feature_maps:
                batch_dict[feature] = batch_dict[feature][inside_mask]
                for mlp_index in range(cfg["patch_size"] ** 2):
                    batch_dict[f"{feature}_{mlp_index}"] = batch_dict[f"{feature}_{mlp_index}"][inside_mask]
            for feature in ["rays_o", "rays_d"]:
                batch_dict[feature] = batch_dict[feature][inside_mask]

            # render each ray for depth, sensor uncertainty, model uncertainty, and color
            ret = self.renderer.render_batch_ray(
                c,
                self.decoders,
                batch_dict["rays_d"],
                batch_dict["rays_o"],
                batch_dict,
                device,
                self.stage,
            )

            depth, var, uncertainty, color = ret

            loss = 0

            for i in range(cfg["num_sensors"]):
                prefix = self.prefix_keys[i]

                depth_mask = batch_dict[f"{prefix}depth"] > 0

                if self.stage == "middle":
                    loss += torch.abs(batch_dict[f"{prefix}depth"][depth_mask] - depth[depth_mask]).sum()
                else:
                    # loss += torch.abs(
                    #     batch_dict[f"{prefix}depth"][depth_mask] - depth[depth_mask]
                    # ).sum()
                    loss += (
                        torch.abs(batch_dict[f"{prefix}depth"][depth_mask] - depth[depth_mask])
                        / (1e-3 + var[depth_mask, i])
                        + torch.log(1e-3 + var[depth_mask, i])
                    ).sum()

            # learn uncertainty in fine stage, refine in color stage
            if self.use_color_in_mapping and self.stage == "fine":
                color_loss = (
                    torch.abs(batch_dict["color"] - color.detach()).sum(dim=1) / (1e-3 + var[..., -1])
                    + torch.log(1e-3 + var[..., -1])
                ).sum()
                loss += color_loss

            if self.use_color_in_mapping and self.stage == "color":
                color_loss = (
                    torch.abs(batch_dict["color"] - color).sum(dim=1) / (1e-3 + var[..., -1])
                    + torch.log(1e-3 + var[..., -1])
                ).sum()
                loss += color_loss
                # color_loss = (torch.abs(batch_dict["color"] - color)).sum()
                # loss += self.w_color_loss * color_loss

            loss.backward(retain_graph=False)
            optimizer.step()
            optimizer.zero_grad()

            # put selected and updated features back to the grid
            if self.frustum_feature_selection:
                for key, val in c.items():
                    val_grad = masked_c_grad[key]
                    mask = masked_c_grad[key + "mask"]
                    val = val.detach()
                    val[mask] = val_grad.clone().detach()
                    c[key] = val

        if self.BA:
            # put the updated camera poses back
            camera_tensor_id = 0
            for id, frame in enumerate(optimize_frame):
                if frame != -1:
                    if frame != oldest_frame:
                        c2w = get_camera_from_tensor(camera_tensor_list[camera_tensor_id].detach())
                        c2w = torch.cat([c2w, bottom], dim=0)
                        camera_tensor_id += 1
                        keyframe_dict[frame]["est_c2w"] = c2w.clone()
                else:
                    c2w = get_camera_from_tensor(camera_tensor_list[-1].detach())
                    c2w = torch.cat([c2w, bottom], dim=0)
                    cur_c2w = c2w.clone()
        if self.BA:
            return cur_c2w
        else:
            return None

    def run(self):
        """Run the mapping module."""

        setup_seed(142857)

        cfg = self.cfg
        frame_dict = self.frame_reader[0]

        self.estimate_c2w_list[0] = frame_dict["gt_c2w"].cpu()
        init = True
        prev_idx = -1

        # mapping loop
        while True:
            # wait for tracking for synchronization
            while True:
                idx = self.idx[0].clone()

                if idx == self.n_img - 1:
                    break
                if self.sync_method == "strict":
                    if idx % self.every_frame == 0 and idx != prev_idx:
                        break

                elif self.sync_method == "loose":
                    if idx == 0 or idx >= prev_idx + self.every_frame // 2:
                        break
                elif self.sync_method == "free":
                    break
                time.sleep(0.1)

            prev_idx = idx
            if self.verbose:
                print(Fore.GREEN)
                print(f"Mapping Frame {idx.item()}")
                print(Style.RESET_ALL)

            frame_dict = self.frame_reader[idx]

            if not init:
                lr_factor = cfg["mapping"]["lr_factor"]
                num_joint_iters = cfg["mapping"]["iters"]

                # here provides a color refinement postprocess
                if idx == self.n_img - 1 and self.color_refine:
                    outer_joint_iters = 5
                    self.mapping_window_size *= 2
                    self.middle_iter_ratio = 0.0
                    self.fine_iter_ratio = 0.0
                    num_joint_iters *= 5
                    self.fix_color = True
                    self.frustum_feature_selection = False
                else:
                    outer_joint_iters = 1

            else:
                outer_joint_iters = 1
                lr_factor = cfg["mapping"]["lr_first_factor"]
                num_joint_iters = cfg["mapping"]["iters_first"]

            # update with estimated camera pose
            cur_c2w = self.estimate_c2w_list[idx].to(self.device)
            frame_dict["est_c2w"] = cur_c2w

            # copy images into modifiable dictionary
            optim_frame_dict = {}

            for key in frame_dict:
                optim_frame_dict[key] = frame_dict[key].clone()

            # set number of points for each frame in the keyframe
            num_joint_iters = num_joint_iters // outer_joint_iters

            # optimize map and add keyframes if applicable
            for outer_joint_iter in range(outer_joint_iters):
                self.BA = (len(self.keyframe_list) > 4) and cfg["mapping"]["BA"]

                _ = self.optimize_map(
                    num_joint_iters,
                    lr_factor,
                    idx,
                    optim_frame_dict,
                    self.keyframe_dict,
                    self.keyframe_list,
                )
                if self.BA:
                    cur_c2w = _
                    self.estimate_c2w_list[idx] = cur_c2w

                # add new frame to keyframe set
                if outer_joint_iter == outer_joint_iters - 1:
                    if (idx % self.keyframe_every == 0 or (idx == self.n_img - 2)) and (idx not in self.keyframe_list):
                        self.keyframe_list.append(idx)
                        result_dict = {}
                        result_dict.update(
                            {
                                "gt_c2w": frame_dict["gt_c2w"].cpu(),
                                "idx": idx,
                                "local": frame_dict["local"].cpu(),
                                "est_c2w": cur_c2w.clone(),
                                "color": frame_dict["color"].cpu(),
                                "gt_depth": frame_dict["gt_depth"].cpu(),
                            }
                        )
                        for i in range(self.cfg["num_sensors"]):
                            prefix = self.prefix_keys[i]
                            result_dict.update(
                                {
                                    f"{prefix}depth": frame_dict[f"{prefix}depth"].cpu(),
                                    f"{prefix}normal": frame_dict[f"{prefix}normal"].cpu(),
                                    f"{prefix}error": frame_dict[f"{prefix}error"].cpu(),
                                    f"{prefix}angle": frame_dict[f"{prefix}angle"].cpu(),
                                    f"{prefix}dx": frame_dict[f"{prefix}error"].cpu(),
                                    f"{prefix}dy": frame_dict[f"{prefix}angle"].cpu(),
                                }
                            )
                        self.keyframe_dict.append(result_dict)

            if self.low_gpu_mem:
                torch.cuda.empty_cache()

            init = False
            # mapping of first frame is done, can begin tracking
            self.mapping_first_frame[0] = 1

            first_frame_dict = self.frame_reader[1]
            first_frame_dict["est_c2w"] = first_frame_dict["gt_c2w"]

            # log and render meshes
            if ((not (idx == 0 and self.no_log_on_first_frame)) and idx % self.ckpt_freq == 0) or idx == self.n_img - 1:
                self.logger.log(
                    idx,
                    self.keyframe_dict,
                    self.keyframe_list,
                    selected_keyframes=self.selected_keyframes if self.save_selected_keyframes_info else None,
                )

            self.mapping_idx[0] = idx
            self.mapping_cnt[0] += 1

            if (idx % self.mesh_freq == 0) and (not (idx == 0 and self.no_mesh_on_first_frame)):
                mesh_out_file = f"{self.output}/mesh/{idx:05d}_mesh.ply"
                self.mesher.get_mesh(
                    mesh_out_file,
                    self.c,
                    self.decoders,
                    self.keyframe_dict,
                    self.estimate_c2w_list,
                    idx,
                    self.device,
                    clean_mesh=self.clean_mesh,
                    get_mask_use_all_frames=False,
                )

            if idx == self.n_img - 1:
                mesh_out_file = f"{self.output}/mesh/final_mesh.ply"
                self.mesher.get_mesh(
                    mesh_out_file,
                    self.c,
                    self.decoders,
                    self.keyframe_dict,
                    self.estimate_c2w_list,
                    idx,
                    self.device,
                    clean_mesh=self.clean_mesh,
                    get_mask_use_all_frames=False,
                )
                os.system(f"cp {mesh_out_file} {self.output}/mesh/{idx:05d}_mesh.ply")
                if self.eval_rec:
                    mesh_out_file = f"{self.output}/mesh/final_mesh_eval_rec.ply"
                    self.mesher.get_mesh(
                        mesh_out_file,
                        self.c,
                        self.decoders,
                        self.keyframe_dict,
                        self.estimate_c2w_list,
                        idx,
                        self.device,
                        show_forecast=False,
                        clean_mesh=self.clean_mesh,
                        get_mask_use_all_frames=True,
                    )
                break

            if idx == self.n_img - 1:
                break
