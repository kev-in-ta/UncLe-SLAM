import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.common import get_camera_from_tensor


class Visualizer(object):
    """
    Visualize intermediate results, render out depth, color and depth uncertainty images.
    It can be called per iteration, which is good for debugging (to see how each tracking/mapping iteration performs).
    NOTE: Only supports upto 2 sensors!
    """

    def __init__(self, freq, inside_freq, vis_dir, renderer, verbose, device="cuda:0"):
        self.freq = freq
        self.device = device
        self.vis_dir = vis_dir
        self.verbose = verbose
        self.renderer = renderer
        self.inside_freq = inside_freq
        os.makedirs(f"{vis_dir}", exist_ok=True)

    def vis(self, idx, iter, frame_dict, feature_maps, c, decoders):
        """Visualize depth, color images and save to file.

        Args:
            idx (int): current frame index.
            iter (int): the iteration number.
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
            feature_maps (List): list of feature maps for rendering.
            c (dicts): feature grids.
            decoders (nn.module): decoders.
            cur_gt_depth, cur_norm, cur_gt_color, cur_gt_error, cur_c2w
        """

        with torch.no_grad():
            if (idx % self.freq == 0) and (iter % self.inside_freq == 0):
                # print(idx, iter)

                if len(frame_dict["est_c2w"].shape) == 1:
                    bottom = (
                        torch.from_numpy(np.array([0, 0, 0, 1.0]).reshape([1, 4])).type(torch.float32).to(self.device)
                    )
                    c2w = get_camera_from_tensor(frame_dict["est_c2w"].clone().detach())
                    c2w = torch.cat([c2w, bottom], dim=0)
                else:
                    c2w = frame_dict["est_c2w"]

                depth, d_var, uncertainty, color = self.renderer.render_img(
                    c,
                    decoders,
                    c2w,
                    frame_dict,
                    feature_maps,
                    self.device,
                    stage="color",
                )

                vis_dict = {}

                for feature in feature_maps:
                    vis_dict[feature] = frame_dict[feature].cpu().numpy()

                depth_np = depth.detach().cpu().numpy()
                color_np = color.detach().cpu().numpy()
                d_var_np = d_var.detach().cpu().numpy()

                h, w = depth_np.shape

                uncertainty_np = uncertainty.detach().cpu().numpy()
                depth_residual = np.abs(vis_dict["depth"] - depth_np)
                depth_residual[vis_dict["depth"] == 0.0] = 0.0
                if "2_depth" in vis_dict:
                    depth_residual2 = np.abs(vis_dict["2_depth"] - depth_np)
                    depth_residual2[vis_dict["2_depth"] == 0.0] = 0.0
                    depth2 = vis_dict["2_depth"]
                    error2 = vis_dict["2_error"]
                    var2 = d_var_np[..., 1]
                else:
                    depth_residual2 = np.zeros((h, w))
                    depth2 = np.zeros((h, w))
                    error2 = np.zeros((h, w))
                    var2 = np.zeros((h, w))
                color_residual = np.abs(vis_dict["color"] - color_np)
                color_residual[vis_dict["depth"] == 0.0] = 0.0

                fig, axs = plt.subplots(3, 5, figsize=(15, 9 * h / w), dpi=300)
                fig.tight_layout()
                max_depth = 5  # np.max(gt_depth_np)
                axs[0, 0].imshow(vis_dict["depth"], cmap="plasma", vmin=0, vmax=max_depth)
                axs[0, 0].set_title("Input Depth 1")
                axs[0, 0].set_xticks([])
                axs[0, 0].set_yticks([])

                axs[0, 1].imshow(depth2, cmap="plasma", vmin=0, vmax=max_depth)
                axs[0, 1].set_title("Input Depth 2")
                axs[0, 1].set_xticks([])
                axs[0, 1].set_yticks([])

                axs[0, 2].imshow(depth_np, cmap="plasma", vmin=0, vmax=max_depth)
                axs[0, 2].set_title("Generated Depth")
                axs[0, 2].set_xticks([])
                axs[0, 2].set_yticks([])

                axs[0, 3].imshow(depth_residual, cmap="plasma", vmin=0, vmax=max_depth)
                axs[0, 3].set_title("Depth Residual 1")
                axs[0, 3].set_xticks([])
                axs[0, 3].set_yticks([])

                axs[0, 4].imshow(depth_residual2, cmap="plasma", vmin=0, vmax=max_depth)
                axs[0, 4].set_title("Depth Residual 2")
                axs[0, 4].set_xticks([])
                axs[0, 4].set_yticks([])

                gt_color_np = np.clip(vis_dict["color"], 0, 1)
                color_np = np.clip(color_np, 0, 1)
                color_residual = np.clip(color_residual, 0, 1)

                axs[1, 0].imshow(np.zeros((h, w)))
                axs[1, 0].set_xticks([])
                axs[1, 0].set_yticks([])

                axs[1, 1].imshow(gt_color_np, cmap="plasma")
                axs[1, 1].set_title("Input RGB")
                axs[1, 1].set_xticks([])
                axs[1, 1].set_yticks([])

                axs[1, 2].imshow(color_np, cmap="plasma")
                axs[1, 2].set_title("Generated RGB")
                axs[1, 2].set_xticks([])
                axs[1, 2].set_yticks([])

                axs[1, 3].imshow(color_residual, cmap="plasma")
                axs[1, 3].set_title("RGB Residual")
                axs[1, 3].set_xticks([])
                axs[1, 3].set_yticks([])

                axs[1, 4].imshow(d_var_np[..., -1], cmap="jet", vmin=0)
                axs[1, 4].set_title(f"Colour Aleatoric: {d_var_np[...,-1].max():.4f}")
                axs[1, 4].set_xticks([])
                axs[1, 4].set_yticks([])

                axs[2, 0].imshow(vis_dict["error"], cmap="jet", vmin=0, vmax=0.2)
                axs[2, 0].set_title(f'GT Difference 1: {vis_dict["error"].max():.4f}')
                axs[2, 0].set_xticks([])
                axs[2, 0].set_yticks([])

                axs[2, 1].imshow(error2, cmap="jet", vmin=0, vmax=0.2)
                axs[2, 1].set_title(f"GT Difference 2: {error2.max():.4f}")
                axs[2, 1].set_xticks([])
                axs[2, 1].set_yticks([])

                axs[2, 2].imshow(np.sqrt(uncertainty_np), cmap="jet", vmin=0, vmax=0.2)
                axs[2, 2].set_title(f"Epistemic: {np.sqrt(uncertainty_np).max():.4f}")
                axs[2, 2].set_xticks([])
                axs[2, 2].set_yticks([])

                axs[2, 3].imshow(d_var_np[..., 0], cmap="jet", vmin=0, vmax=0.2)
                axs[2, 3].set_title(f"Aleatoric 1: {d_var_np[...,0].max():.4f}")
                axs[2, 3].set_xticks([])
                axs[2, 3].set_yticks([])

                axs[2, 4].imshow(var2, cmap="jet", vmin=0, vmax=0.2)
                axs[2, 4].set_title(f"Aleatoric 2: {var2.max():.4f}")
                axs[2, 4].set_xticks([])
                axs[2, 4].set_yticks([])

                plt.subplots_adjust(wspace=0.03, hspace=0.20)

                plt.savefig(
                    f"{self.vis_dir}/{idx:05d}_{iter:04d}.jpg",
                    bbox_inches="tight",
                    pad_inches=0.2,
                )
                plt.clf()

                uncertainty_np = np.sqrt(uncertainty_np)

                plt.imsave(
                    f"{self.vis_dir}/proxy1_{idx:05d}.png",
                    vis_dict["error"],
                    cmap="jet",
                    vmin=0,
                    vmax=0.2,
                )
                plt.imsave(
                    f"{self.vis_dir}/proxy2_{idx:05d}.png",
                    error2,
                    cmap="jet",
                    vmin=0,
                    vmax=0.2,
                )
                plt.imsave(
                    f"{self.vis_dir}/deep1_{idx:05d}.png",
                    d_var_np[..., 0],
                    cmap="jet",
                    vmin=0,
                    vmax=0.2,
                )
                plt.imsave(
                    f"{self.vis_dir}/deep2_{idx:05d}.png",
                    var2,
                    cmap="jet",
                    vmin=0,
                    vmax=0.2,
                )
                plt.imsave(
                    f"{self.vis_dir}/model_{idx:05d}.png",
                    uncertainty_np,
                    cmap="jet",
                    vmin=0,
                    vmax=0.2,
                )
                plt.imsave(
                    f"{self.vis_dir}/render_depth_{idx:05d}.png",
                    depth_np,
                    cmap="plasma",
                    vmin=0,
                    vmax=max_depth,
                )
                plt.imsave(
                    f"{self.vis_dir}/gt_depth_{idx:05d}.png",
                    vis_dict["depth"],
                    cmap="plasma",
                    vmin=0,
                    vmax=max_depth,
                )

                if self.verbose:
                    print(
                        f"Saved rendering visualization of color/depth image at {self.vis_dir}/{idx:05d}_{iter:04d}.jpg"
                    )
