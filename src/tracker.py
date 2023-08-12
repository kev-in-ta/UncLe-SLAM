import copy
import os
import time

import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import get_camera_from_tensor, get_samples, get_tensor_from_camera
from src.utils.datasets import get_dataset
from src.utils.visualizer import Visualizer


def setup_seed(seed):
    """Set manual seed for various libraries."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class Tracker(object):
    """
    Tracker class that updates pose information from most recent map.

    """

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

        self.scale = cfg["scale"]
        self.occupancy = cfg["occupancy"]
        self.sync_method = cfg["sync_method"]

        self.idx = slam.idx
        self.bound = slam.bound
        self.mesher = slam.mesher
        self.output = slam.output
        self.verbose = slam.verbose
        self.shared_c = slam.shared_c
        self.renderer = slam.renderer
        self.gt_c2w_list = slam.gt_c2w_list
        self.low_gpu_mem = slam.low_gpu_mem
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.shared_decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list

        self.cam_lr = cfg["tracking"]["lr"]
        self.device = cfg["tracking"]["device"]
        self.num_cam_iters = cfg["tracking"]["iters"]
        self.gt_camera = cfg["tracking"]["gt_camera"]
        self.tracking_pixels = cfg["tracking"]["pixels"]
        self.seperate_LR = cfg["tracking"]["seperate_LR"]
        self.w_color_loss = cfg["tracking"]["w_color_loss"]
        self.ignore_edge_W = cfg["tracking"]["ignore_edge_W"]
        self.ignore_edge_H = cfg["tracking"]["ignore_edge_H"]
        self.handle_dynamic = cfg["tracking"]["handle_dynamic"]
        self.use_color_in_tracking = cfg["tracking"]["use_color_in_tracking"]
        self.const_speed_assumption = cfg["tracking"]["const_speed_assumption"]

        self.every_frame = cfg["mapping"]["every_frame"]
        self.no_vis_on_first_frame = cfg["mapping"]["no_vis_on_first_frame"]

        self.prev_mapping_idx = -1
        self.frame_reader = get_dataset(cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, shuffle=False, num_workers=1)
        self.visualizer = Visualizer(
            freq=cfg["tracking"]["vis_freq"],
            inside_freq=cfg["tracking"]["vis_inside_freq"],
            vis_dir=os.path.join(self.output, "vis" if "Demo" in self.output else "tracking_vis"),
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

    def optimize_cam_in_batch(self, frame_dict, batch_size, optimizer):
        """Do one iteration of camera iteration. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            frame_dict: dictionary of images / poses for the current frame.
                gt_c2w (tensor): ground truth pose.
                est_c2w (tensor): estimated pose / camera tensor.
                color (tensor): measured color image of the current frame. (H,W,3)
                Per sensor:
                    depth (tensor): measured depth image of the current frame. (H,W)
                    normal (tensor): smoothed normal map from depth image. (H,W,3)
                    local (tensor): local ray direction in camera reference. (H,W,3)
                    error (tensor): absolute difference GT vs. measured depth (vis only). (H, W)
                    angle (tensor): angle difference between normal and local ray. (H,W)
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.

        Returns:
            loss (float): The value of loss.

        """
        device = self.device
        cfg = self.cfg
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        optimizer.zero_grad()
        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H

        # get estimated pose
        frame_dict["est_c2w"] = get_camera_from_tensor(frame_dict["est_c2w"])

        # copy dictionary maps to pass into renderer for selected features
        optim_frame_dict = {}
        for feature in self.feature_maps + ["est_c2w"]:
            optim_frame_dict[feature] = frame_dict[feature].to(device)

        # get sample ray directions and associated pixels per ray
        batch_rays_o, batch_rays_d, batch_dict = get_samples(
            Hedge,
            H - Hedge,
            Wedge,
            W - Wedge,
            batch_size,
            H,
            W,
            fx,
            fy,
            cx,
            cy,
            optim_frame_dict,
            self.feature_maps,
            self.device,
            kernel=self.cfg["patch_size"],
        )

        batch_dict["rays_o"] = batch_rays_o.float()
        batch_dict["rays_d"] = batch_rays_d.float()

        # should pre-filter those out of bounding box depth value
        with torch.no_grad():
            det_rays_o = batch_dict["rays_o"].clone().detach().unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = batch_dict["rays_d"].clone().detach().unsqueeze(-1)  # (N, 3, 1)
            t = (self.bound.unsqueeze(0).to(device) - det_rays_o) / det_rays_d
            t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            inside_mask = t >= batch_dict["depth"]
        for feature in self.feature_maps:
            batch_dict[feature] = batch_dict[feature][inside_mask]
            for mlp_index in range(self.cfg["patch_size"] ** 2):
                batch_dict[f"{feature}_{mlp_index}"] = batch_dict[f"{feature}_{mlp_index}"][inside_mask]
        for feature in ["rays_o", "rays_d"]:
            batch_dict[feature] = batch_dict[feature][inside_mask]

        # render each ray to get the predicted depth, sensor uncertainty, model uncertainty, and color
        ret = self.renderer.render_batch_ray(
            self.c,
            self.decoders,
            batch_dict["rays_d"],
            batch_dict["rays_o"],
            batch_dict,
            self.device,
            stage="color",
        )
        depth, var, uncertainty, color = ret

        # detach uncertainty to prevent backprop in tracking
        uncertainty = uncertainty.detach()
        var = var.detach()

        for i in range(cfg["num_sensors"]):
            prefix = self.prefix_keys[i]

            # calculate loss, masking out outliers if handling dynamic objects flag is set
            if self.handle_dynamic:
                tmp = torch.abs(batch_dict[f"{prefix}depth"] - depth) / (torch.sqrt(uncertainty + 1e-10) + (var[:, i]))
                mask = (tmp < 10 * tmp.median()) & (batch_dict[f"{prefix}depth"] > 0)
            else:
                mask = batch_dict[f"{prefix}depth"] > 0

            loss = (torch.abs(batch_dict[f"{prefix}depth"] - depth) / (torch.sqrt(uncertainty + 1e-10) + (var[:, i])))[
                mask
            ].sum()

        if self.use_color_in_tracking:
            color_loss = (torch.abs(batch_dict["color"] - color).sum(dim=1) / (1e-3 + var[..., -1]))[mask].sum()
            loss += color_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    def update_para_from_mapping(self):
        """
        Update the parameters of scene representation from the mapping thread.

        """
        if self.mapping_idx[0] != self.prev_mapping_idx:
            if self.verbose:
                print("Tracking: update the parameters from mapping")
            self.decoders = copy.deepcopy(self.shared_decoders).to(self.device)
            for key, val in self.shared_c.items():
                val = val.clone().to(self.device)
                self.c[key] = val
            self.prev_mapping_idx = self.mapping_idx[0].clone()

    def run(self):
        """Run the tracking module."""

        setup_seed(142857)

        device = self.device
        self.c = {}
        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader)

        # loop through each frame
        for frame in pbar:  # idx, gt_c2w, color, depth, normal, local, angle, error
            # get individual frames
            frame_dict = {}

            for feature in frame:
                frame_dict[feature] = frame[feature][0]

            if not self.verbose:
                pbar.set_description(f"Tracking Frame {frame_dict['idx']}")

            # wait for synchronization if applicable
            if self.sync_method == "strict":
                # strictly mapping and then tracking
                # initiate mapping every self.every_frame frames
                if frame_dict["idx"] > 0 and (frame_dict["idx"] % self.every_frame == 1 or self.every_frame == 1):
                    while self.mapping_idx[0] != frame_dict["idx"] - 1:
                        time.sleep(0.1)
                    pre_c2w = self.estimate_c2w_list[frame_dict["idx"] - 1].to(device)
            elif self.sync_method == "loose":
                # mapping idx can be later than tracking idx is within the bound of
                # [-self.every_frame-self.every_frame//2, -self.every_frame+self.every_frame//2]
                while self.mapping_idx[0] < frame_dict["idx"] - self.every_frame - self.every_frame // 2:
                    time.sleep(0.1)
            elif self.sync_method == "free":
                # pure parallel, if mesh/vis happens may cause inbalance
                pass

            # get newest mapping information
            self.update_para_from_mapping()

            if self.verbose:
                print(Fore.MAGENTA)
                print("Tracking Frame ", frame_dict["idx"].item())
                print(Style.RESET_ALL)

            # anchor the tracking with the initial frame
            if frame_dict["idx"] == 0 or self.gt_camera:
                frame_dict["est_c2w"] = frame_dict["gt_c2w"]
                if not self.no_vis_on_first_frame:
                    self.visualizer.vis(
                        frame_dict["idx"],
                        0,
                        frame_dict,
                        self.feature_maps,
                        self.c,
                        self.decoders,
                    )
            # else perform tracking optimization
            else:
                gt_camera_tensor = get_tensor_from_camera(frame_dict["gt_c2w"])
                if self.const_speed_assumption and frame_dict["idx"] - 2 >= 0:
                    pre_c2w = pre_c2w.float()
                    delta = pre_c2w @ self.estimate_c2w_list[frame_dict["idx"] - 2].to(device).float().inverse()
                    estimated_new_cam_c2w = delta @ pre_c2w
                else:
                    estimated_new_cam_c2w = pre_c2w

                camera_tensor = get_tensor_from_camera(estimated_new_cam_c2w.detach())
                if self.seperate_LR:
                    camera_tensor = camera_tensor.to(device).detach()
                    T = camera_tensor[-3:]
                    quad = camera_tensor[:4]
                    cam_para_list_quad = [quad]
                    quad = Variable(quad, requires_grad=True)
                    T = Variable(T, requires_grad=True)
                    camera_tensor = torch.cat([quad, T], 0)
                    cam_para_list_T = [T]
                    cam_para_list_quad = [quad]
                    optimizer_camera = torch.optim.Adam(
                        [
                            {"params": cam_para_list_T, "lr": self.cam_lr},
                            {"params": cam_para_list_quad, "lr": self.cam_lr * 0.2},
                        ]
                    )
                else:
                    camera_tensor = Variable(camera_tensor.to(device), requires_grad=True)
                    cam_para_list = [camera_tensor]
                    optimizer_camera = torch.optim.Adam(cam_para_list, lr=self.cam_lr)

                initial_loss_camera_tensor = torch.abs(gt_camera_tensor.to(device) - camera_tensor).mean().item()
                candidate_cam_tensor = None
                current_min_loss = 10000000000.0

                # optimize loss over # tracking iterations
                for cam_iter in range(self.num_cam_iters):
                    if self.seperate_LR:
                        camera_tensor = torch.cat([quad, T], 0).to(self.device)

                    frame_dict["est_c2w"] = camera_tensor

                    self.visualizer.vis(
                        frame_dict["idx"],
                        cam_iter,
                        frame_dict,
                        self.feature_maps,
                        self.c,
                        self.decoders,
                    )

                    loss = self.optimize_cam_in_batch(frame_dict, self.tracking_pixels, optimizer_camera)

                    if cam_iter == 0:
                        initial_loss = loss

                    loss_camera_tensor = torch.abs(gt_camera_tensor.to(device) - camera_tensor).mean().item()
                    if self.verbose:
                        if cam_iter == self.num_cam_iters - 1:
                            print(
                                f"Re-rendering loss: {initial_loss:.2f}->{loss:.2f} "
                                + f"camera tensor error: {initial_loss_camera_tensor:.4f}->{loss_camera_tensor:.4f}"
                            )
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_tensor = camera_tensor.clone().detach()
                bottom = torch.from_numpy(np.array([0, 0, 0, 1.0]).reshape([1, 4])).type(torch.float32).to(self.device)
                c2w = get_camera_from_tensor(candidate_cam_tensor.clone().detach())
                c2w = torch.cat([c2w, bottom], dim=0)
                frame_dict["est_c2w"] = c2w
            self.estimate_c2w_list[frame_dict["idx"]] = frame_dict["est_c2w"].clone().cpu()
            self.gt_c2w_list[frame_dict["idx"]] = frame_dict["gt_c2w"].clone().cpu()
            pre_c2w = frame_dict["est_c2w"].clone()
            self.idx[0] = frame_dict["idx"]
            if self.low_gpu_mem:
                torch.cuda.empty_cache()
