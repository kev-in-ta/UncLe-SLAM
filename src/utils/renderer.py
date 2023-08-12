import torch

from src.common import get_rays, raw2outputs_nerf_color, sample_pdf


class Renderer(object):
    def __init__(self, cfg, args, slam, points_batch_size=500000, ray_batch_size=50000):
        self.prefix_keys = ["", "2_", "3_"]
        self.num_sensors = cfg["num_sensors"]
        self.num_features = cfg["num_features"]

        self.patch_size = cfg["patch_size"]
        self.unused_list = ["error", "rays_o", "rays_d", "color", "local"]
        if self.num_features == 2:
            self.unused_list += ["dx", "dy", "normal"]
        elif self.num_features == 7:
            pass
        else:
            raise Exception("Number of features != 2 or 7")

        self.ray_batch_size = ray_batch_size
        self.points_batch_size = points_batch_size

        self.lindisp = cfg["rendering"]["lindisp"]
        self.perturb = cfg["rendering"]["perturb"]
        self.N_samples = cfg["rendering"]["N_samples"]
        self.N_surface = cfg["rendering"]["N_surface"]
        self.N_importance = cfg["rendering"]["N_importance"]

        self.scale = cfg["scale"]
        self.occupancy = cfg["occupancy"]
        self.bound = slam.bound

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = (
            slam.H,
            slam.W,
            slam.fx,
            slam.fy,
            slam.cx,
            slam.cy,
        )

    def eval_points(self, p, d_feats, c_feats, decoders, c=None, stage="color", device="cuda:0"):
        """Evaluate the occupancy and/or color value for the points.

        Args:
            p (tensor, N*3): Point coordinates.
            d_feats (tensor, N*Y*2): Tensor of added features for uncertainty evaluation.
            c_feats (tensor, N*Z*2): Tensor of added features for color uncertainty evaluation.
            decoders (nn.module decoders): Decoders.
            c (dicts, optional): Feature grids. Defaults to None.
            stage (str, optional): Query stage, corresponds to different levels. Defaults to 'color'.
            device (str, optional): CUDA device. Defaults to 'cuda:0'.

        Returns:
            ret (tensor): occupancy (and color) value of input points.
        """

        p_split = torch.split(p, self.points_batch_size)

        bound = self.bound

        rets = []

        for i, pi in enumerate(p_split):
            # mask for points out of bound
            mask_x = (pi[:, 0] < bound[0][1]) & (pi[:, 0] > bound[0][0])
            mask_y = (pi[:, 1] < bound[1][1]) & (pi[:, 1] > bound[1][0])
            mask_z = (pi[:, 2] < bound[2][1]) & (pi[:, 2] > bound[2][0])
            mask = mask_x & mask_y & mask_z

            pi = pi.unsqueeze(0)
            ret, var = decoders(pi, c_grid=c, stage=stage, d_feats=d_feats, c_feats=c_feats)
            ret = ret.squeeze(0)
            if len(ret.shape) == 1 and ret.shape[0] == 4:
                ret = ret.unsqueeze(0)

            ret[~mask, 3] = 100
            rets.append(ret)

        ret = torch.cat(rets, dim=0)

        return ret, var

    def render_batch_ray(self, c, decoders, rays_d, rays_o, batch_dict, device, stage):
        """Render color, depth and uncertainty of a batch of rays.

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            rays_d (tensor, N*3): rays direction.
            rays_o (tensor, N*3): rays origin.
            batch_dict (dict): Includes various features. sensor depth image, normal, color...
            device (str): device name to compute on.
            stage (str): query stage.

        Returns:
            depth (tensor): rendered depth.
            d_var (tensor): rendered sensor uncertainty.
            uncertainty (tensor): rendered model uncertainty.
            color (tensor): rendered color.

        """

        N_samples = self.N_samples
        N_surface = self.N_surface
        N_importance = self.N_importance

        N_rays = rays_o.shape[0]

        features = list(batch_dict.keys())

        c_features = [feat for feat in features if (feat[-1].isdigit() and feat.startswith("color"))]

        # only keeps expanded patch of features
        features = [
            feat for feat in features if (feat[-1].isdigit() and not any(feat.startswith(s) for s in self.unused_list))
        ]

        # filters out additional sensors
        features = [feat for feat in features if not feat.startswith("2")]

        gt_depth_samples = batch_dict["depth"].reshape(-1, 1).repeat(1, N_samples)
        for i in range(1, self.num_sensors):
            prefix = self.prefix_keys[i]
            gt_depth_samples += batch_dict[f"{prefix}depth"].reshape(-1, 1).repeat(1, N_samples)
        gt_depth_samples /= self.num_sensors
        near = gt_depth_samples * 0.01

        with torch.no_grad():
            det_rays_o = rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            t = (self.bound.unsqueeze(0).to(device) - det_rays_o) / det_rays_d  # (N, 3, 2)
            far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            far_bb = far_bb.unsqueeze(-1)
            far_bb += 0.01

        # in case the bound is too large
        max_depth = torch.max(batch_dict["depth"] * 1.2)
        for i in range(1, self.num_sensors):
            prefix = self.prefix_keys[i]
            max_depth = max((max_depth, torch.max(batch_dict[f"{prefix}depth"] * 1.2)))
        far = torch.clamp(far_bb, 0, max_depth)  # NOTE: may cause vis to be incomplete if all depth measurements are 0
        if far.mean() == 0:
            far = torch.ones_like(far) * far_bb

        # add points selected near surface
        if N_surface > 0:
            # since we want to colorize even on regions with no depth sensor readings,
            # meaning colorize on interpolated geometry region,
            # we sample all pixels (not using depth mask) for color loss.
            # Therefore, for pixels with non-zero depth value, we sample near the surface,
            # since it is not a good idea to sample 16 points near (half even behind) camera,
            # for pixels with zero depth value, we sample uniformly from camera to max_depth.

            z_val_surface_list = []

            for i in range(self.num_sensors):
                depth_key = f"{self.prefix_keys[i]}depth"
                gt_none_zero_mask = batch_dict[depth_key] > 0
                gt_none_zero = batch_dict[depth_key][gt_none_zero_mask]
                gt_none_zero = gt_none_zero.unsqueeze(-1)
                gt_depth_surface = gt_none_zero.repeat(1, N_surface)
                t_vals_surface = torch.linspace(0.0, 1.0, steps=N_surface).double().to(device)
                # emperical range 0.05*depth
                z_vals_surface_depth_none_zero = 0.95 * gt_depth_surface * (
                    1.0 - t_vals_surface
                ) + 1.05 * gt_depth_surface * (t_vals_surface)
                z_vals_surface = torch.zeros(batch_dict[depth_key].shape[0], N_surface).to(device).double()
                gt_none_zero_mask = gt_none_zero_mask.squeeze(-1)
                z_vals_surface[gt_none_zero_mask, :] = z_vals_surface_depth_none_zero
                near_surface = 0.001
                far_surface = torch.max(batch_dict[depth_key])
                z_vals_surface_depth_zero = near_surface * (1.0 - t_vals_surface) + far_surface * (t_vals_surface)

                z_vals_surface_depth_zero = z_vals_surface_depth_zero.unsqueeze(0).repeat((~gt_none_zero_mask).sum(), 1)

                z_vals_surface[~gt_none_zero_mask, :] = z_vals_surface_depth_zero

                z_val_surface_list.append(z_vals_surface)

            z_vals_surface = torch.cat(z_val_surface_list, -1)

        t_vals = torch.linspace(0.0, 1.0, steps=N_samples, device=device)

        if not self.lindisp:
            z_vals = near * (1.0 - t_vals) + far * (t_vals)
        else:
            z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * (t_vals))

        if self.perturb > 0.0:
            # get intervals between samples
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)
            z_vals = lower + (upper - lower) * t_rand

        if N_surface > 0:
            z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_surface.double()], -1), -1)

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples+N_surface, 3]

        pointsf = pts.reshape(-1, 3)

        features_list = []
        c_features_list = []
        c_num_features = 0

        for i in range(self.num_sensors):
            features_list.append([])
            prefix = self.prefix_keys[i]
            num_features = 0

            for feature in features:
                if batch_dict[feature].shape[-1] == 3:
                    num_features += 3
                    features_list[-1].append(batch_dict[f"{prefix}{feature}"].reshape(-1, 3))
                else:
                    num_features += 1
                    features_list[-1].append(batch_dict[f"{prefix}{feature}"].reshape(-1, 1))

        for feature in c_features:
            c_num_features += 3
            c_features_list.append(batch_dict[feature].reshape(-1, 3))

        d_feats_list = []

        for i in range(self.num_sensors):
            d_feats_list.append(torch.cat(features_list[i], 1).to(device).float())
            d_feats_list[-1] = d_feats_list[-1].reshape(-1, num_features)

        d_feats = torch.cat([feats.unsqueeze(2) for feats in d_feats_list], 2)
        c_feats = torch.cat(c_features_list, 1).to(device).float()
        c_feats = c_feats.reshape(-1, c_num_features)

        raw, var = self.eval_points(pointsf, d_feats, c_feats, decoders, c, stage, device)

        raw = raw.reshape(N_rays, N_samples + self.num_sensors * N_surface, -1)

        depth, var_out, uncertainty, color, weights = raw2outputs_nerf_color(
            raw,
            var,
            z_vals,
            rays_d,
            occupancy=self.occupancy,
            device=device,
        )

        if N_importance > 0:
            z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid,
                weights[..., 1:-1],
                N_importance,
                det=(self.perturb == 0.0),
                device=device,
            )
            z_samples = z_samples.detach()
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
            pts = pts.reshape(-1, 3)
            raw, var = self.eval_points(pts, decoders, c, stage, device)
            raw = raw.reshape(N_rays, N_samples + N_importance + N_surface, -1)

            depth, uncertainty, color, weights = raw2outputs_nerf_color(
                raw, var, z_vals, rays_d, occupancy=self.occupancy, device=device
            )

        return depth, var_out, uncertainty, color

    def render_img(self, c, decoders, c2w, frame_dict, feature_maps, device, stage):
        """Render out depth, uncertainty, and color images.

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            c2w (tensor): camera to world matrix of current frame.
            device (str): device name to compute on.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None.

        Returns:
            depth (tensor, H*W): rendered depth image.
            uncertainty (tensor, H*W): rendered uncertainty image.
            color (tensor, H*W*3): rendered color image.
        """

        img_dict = {}

        with torch.no_grad():
            H = self.H
            W = self.W

            rays_o, rays_d = get_rays(H, W, self.fx, self.fy, self.cx, self.cy, c2w, device)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            depth_list = []
            d_var_list = []
            uncertainty_list = []
            color_list = []

            ray_batch_size = self.ray_batch_size

            for feature in feature_maps:
                if frame_dict[feature].shape[-1] == 3:
                    img_dict[feature] = frame_dict[feature].reshape(-1, 3)
                else:
                    img_dict[feature] = frame_dict[feature].reshape(-1)
                for n in range(self.patch_size):
                    for m in range(self.patch_size):
                        mlp_index = n * self.patch_size + m
                        temp_feature_map = torch.roll(
                            frame_dict[feature],
                            shifts=(n - self.patch_size // 2, m - self.patch_size // 2),
                            dims=(0, 1),
                        )
                        if frame_dict[feature].shape[-1] == 3:
                            img_dict[f"{feature}_{mlp_index}"] = temp_feature_map.reshape(-1, 3)
                        else:
                            img_dict[f"{feature}_{mlp_index}"] = temp_feature_map.reshape(-1)

            img_dict["rays_o"] = rays_o.float()
            img_dict["rays_d"] = rays_d.float()

            batch_dict = {}

            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i : i + ray_batch_size]
                rays_o_batch = rays_o[i : i + ray_batch_size]

                for feature in feature_maps + ["rays_o", "rays_d"]:
                    batch_dict[feature] = img_dict[feature][i : i + ray_batch_size]
                    for mlp_index in range(self.patch_size**2):
                        if feature not in ["rays_o", "rays_d"]:
                            batch_dict[f"{feature}_{mlp_index}"] = img_dict[f"{feature}_{mlp_index}"][
                                i : i + ray_batch_size
                            ]
                ret = self.render_batch_ray(c, decoders, rays_d_batch, rays_o_batch, batch_dict, device, stage)

                depth, d_var, uncertainty, color = ret
                depth_list.append(depth.double())
                d_var_list.append(d_var.double())
                uncertainty_list.append(uncertainty.double())
                color_list.append(color)

            depth = torch.cat(depth_list, dim=0)
            d_var = torch.cat(d_var_list, dim=0)
            uncertainty = torch.cat(uncertainty_list, dim=0)
            color = torch.cat(color_list, dim=0)

            depth = depth.reshape(H, W)
            d_var = d_var.reshape(H, W, -1)
            uncertainty = uncertainty.reshape(H, W)
            color = color.reshape(H, W, 3)
            return depth, d_var, uncertainty, color
