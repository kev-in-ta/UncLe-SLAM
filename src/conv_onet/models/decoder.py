import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common import normalize_3d_coordinate


class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    Modified based on the implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    """

    def __init__(self, num_input_channels, mapping_size=93, scale=25, learnable=True):
        super().__init__()

        if learnable:
            self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale)
        else:
            self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        """Forward pass through Gaussian Fourier feature mapping."""
        x = x.squeeze(0)
        assert x.dim() == 2, f"Expected 2D input (got {x.dim()}D input)"
        x = x @ self._B.to(x.device)
        return torch.sin(x)


class NerfPositionalEmbedding(torch.nn.Module):
    """
    Nerf positional embedding.

    """

    def __init__(self, multires, log_sampling=True):
        super().__init__()
        self.log_sampling = log_sampling
        self.include_input = True
        self.periodic_fns = [torch.sin, torch.cos]
        self.max_freq_log2 = multires - 1
        self.num_freqs = multires
        self.max_freq = self.max_freq_log2
        self.N_freqs = self.num_freqs

    def forward(self, x):
        """Forward pass through NeRF positional embedding."""
        x = x.squeeze(0)
        assert x.dim() == 2, f"Expected 2D input (got {x.dim()}D input)"

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, self.max_freq, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**self.max_freq, steps=self.N_freqs)
        output = []
        if self.include_input:
            output.append(x)
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                output.append(p_fn(x * freq))
        ret = torch.cat(output, dim=1)
        return ret


class DenseLayer(nn.Linear):
    """Single dense layer for MLP."""

    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        """Resets dense layer parameters."""
        torch.nn.init.xavier_uniform_(self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class Same(nn.Module):
    """No encoding layer, returns inputs."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Returns flattened inputs."""
        x = x.squeeze(0)
        return x


class MLP(nn.Module):
    """
    Decoder. Point coordinates not only used in sampling the feature grids, but also as MLP input.

    Args:
        name (str): name of this decoder.
        dim (int): input dimension.
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of Decoder network.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
        sample_mode (str): sampling feature strategy, bilinear|nearest.
        color (bool): whether or not to output color.
        skips (list): list of layers to have skip connections.
        grid_len (float): voxel length of its corresponding feature grid.
        pos_embedding_method (str): positional embedding method.
        concat_feature (bool): whether to get feature from middle level and concat to the current feature.
        conf_dim (int): number of dimensions for ray dependent input.
    """

    def __init__(
        self,
        name="",
        dim=3,
        c_dim=128,
        hidden_size=256,
        n_blocks=5,
        leaky=False,
        sample_mode="bilinear",
        color=False,
        uncertainty=False,
        skips=[2],
        grid_len=0.16,
        pos_embedding_method="fourier",
        concat_feature=False,
        conf_dim=7,
    ):
        super().__init__()

        self.name = name
        self.color = color
        self.uncertainty = uncertainty
        self.no_grad_feature = False
        self.c_dim = c_dim
        self.grid_len = grid_len
        self.concat_feature = concat_feature
        self.n_blocks = n_blocks
        self.skips = skips

        # add appended 2D feature map information
        if self.uncertainty:
            add_features = conf_dim
        else:
            add_features = 0

        if c_dim != 0:
            self.fc_c = nn.ModuleList([nn.Linear(c_dim, hidden_size) for i in range(n_blocks)])

        # create positional encoding blocks
        if pos_embedding_method == "fourier":
            embedding_size = 93
            self.embedder = GaussianFourierFeatureTransform(dim, mapping_size=embedding_size, scale=25)
        elif pos_embedding_method == "same":
            embedding_size = 3
            self.embedder = Same()
        elif pos_embedding_method == "nerf":
            if "color" in name:
                multires = 10
                self.embedder = NerfPositionalEmbedding(multires, log_sampling=True)
            else:
                multires = 5
                self.embedder = NerfPositionalEmbedding(multires, log_sampling=False)
            embedding_size = multires * 6 + 3
        elif pos_embedding_method == "fc_relu":
            embedding_size = 93
            self.embedder = DenseLayer(dim, embedding_size, activation="relu")

        # create MLP with skip connections
        self.pts_linears = nn.ModuleList(
            [DenseLayer(embedding_size + add_features, hidden_size, activation="relu")]
            + [
                DenseLayer(hidden_size, hidden_size, activation="relu")
                if i not in self.skips
                else DenseLayer(
                    hidden_size + embedding_size + add_features,
                    hidden_size,
                    activation="relu",
                )
                for i in range(n_blocks - 1)
            ]
        )

        # display network information
        print(self.name)
        print(self.pts_linears)
        print(self.fc_c)

        if self.color:
            self.output_linear = DenseLayer(hidden_size, 4, activation="linear")
        else:
            self.output_linear = DenseLayer(hidden_size, 1, activation="linear")

        if self.uncertainty:
            self.output_softplus = nn.Softplus()

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        p_nor = p_nor.unsqueeze(0)
        vgrid = p_nor[:, :, None, None].float()
        # acutally trilinear interpolation if mode = 'bilinear'
        c = (
            F.grid_sample(
                c,
                vgrid,
                padding_mode="border",
                align_corners=True,
                mode=self.sample_mode,
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        return c

    def forward(self, p, c_grid=None, v=None):
        if self.c_dim != 0:
            c = self.sample_grid_feature(p, c_grid["grid_" + self.name]).transpose(1, 2).squeeze(0)

            if self.concat_feature:
                # only happen to fine decoder, get feature from middle level and concat to the current feature
                with torch.no_grad():
                    c_middle = self.sample_grid_feature(p, c_grid["grid_middle"]).transpose(1, 2).squeeze(0)
                c = torch.cat([c, c_middle], dim=1)

        p = p.float()

        embedded_pts = self.embedder(p)

        if self.uncertainty:
            embedded_pts = torch.cat([embedded_pts, v], dim=1)

        h = embedded_pts

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if self.c_dim != 0:
                h = h + self.fc_c[i](c)
            if i in self.skips:
                h = torch.cat([embedded_pts, h], -1)
        out = self.output_linear(h)
        if self.uncertainty:
            out = self.output_softplus(out)
        if not self.color:
            out = out.squeeze(-1)
        return out


class MLP_no_xyz(nn.Module):
    """
    Decoder. Point coordinates only used in sampling the feature grids, not as MLP input.

    Args:
        name (str): name of this decoder.
        dim (int): input dimension.
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of Decoder network.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
        sample_mode (str): sampling feature strategy, bilinear|nearest.
        color (bool): whether or not to output color.
        skips (list): list of layers to have skip connection.
        grid_len (float): voxel length of its corresponding feature grid.
    """

    def __init__(
        self,
        name="",
        dim=3,
        c_dim=128,
        hidden_size=256,
        n_blocks=5,
        leaky=False,
        sample_mode="bilinear",
        color=False,
        uncertainty=False,
        skips=[2],
        grid_len=0.16,
    ):
        super().__init__()
        self.name = name
        self.no_grad_feature = False
        self.color = color
        self.uncertainty = uncertainty
        self.grid_len = grid_len
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.skips = skips

        self.pts_linears = nn.ModuleList(
            [DenseLayer(hidden_size, hidden_size, activation="relu")]
            + [
                DenseLayer(hidden_size, hidden_size, activation="relu")
                if i not in self.skips
                else DenseLayer(hidden_size + c_dim, hidden_size, activation="relu")
                for i in range(n_blocks - 1)
            ]
        )

        print(self.name)
        print(self.pts_linears)

        if self.color:
            self.output_linear = DenseLayer(hidden_size, 4, activation="linear")
        else:
            self.output_linear = DenseLayer(hidden_size, 1, activation="linear")

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

    def sample_grid_feature(self, p, grid_feature):
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        p_nor = p_nor.unsqueeze(0)
        vgrid = p_nor[:, :, None, None].float()
        c = (
            F.grid_sample(
                grid_feature,
                vgrid,
                padding_mode="border",
                align_corners=True,
                mode=self.sample_mode,
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        return c

    def forward(self, p, c_grid=None, **kwargs):
        c = self.sample_grid_feature(p, c_grid["grid_" + self.name]).transpose(1, 2).squeeze(0)
        h = c
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([c, h], -1)
        out = self.output_linear(h)
        if not self.color:
            out = out.squeeze(-1)
        return out


class MLP_2D(nn.Module):
    """
    Simple MLP that takes in pixel information and outputs a pixel result.

    Args:
        name (str): name of this network.
        dim (int): input dimension.
        hidden_size (int): hidden size of Decoder network.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
    """

    def __init__(self, name="", dim=10, hidden_size=32, n_blocks=5, leaky=False):
        super().__init__()

        self.name = name
        self.no_grad_feature = False
        self.n_blocks = n_blocks

        self.linear = nn.ModuleList(
            [DenseLayer(dim, hidden_size, activation="relu")]
            + [DenseLayer(hidden_size, hidden_size, activation="relu") for i in range(n_blocks - 1)]
        )

        print(self.name)
        print(self.linear)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.output_linear = DenseLayer(hidden_size, 1, activation="linear")

        self.output_softplus = nn.Softplus()
        # self.output_sigmoid = nn.Sigmoid()

    def forward(self, features, **kwargs):
        for i, l in enumerate(self.linear):
            features = self.linear[i](features)
            features = F.relu(features)

        out = self.output_linear(features)
        out = self.output_softplus(out)
        # out = self.output_sigmoid(out)

        return out


class NICE(nn.Module):
    """
    Neural Implicit Scalable Encoding.

    Args:
        dim (int): input dimension.
        conf_dim (int): input dimension for uncertainty network.
        c_dim (int): feature dimension.
        middle_grid_len (float): voxel length in middle grid.
        fine_grid_len (float): voxel length in fine grid.
        color_grid_len (float): voxel length in color grid.
        hidden_size (int): hidden size of decoder network
        pos_embedding_method (str): positional embedding method.
        uncertainty (str): whether to use 2D or 3D architecture.
        num_sensors (int): number of uncertainty heads to intialize for each sensor.
    """

    def __init__(
        self,
        dim=3,
        conf_dim=50,
        col_conf_dim=75,
        c_dim=32,
        middle_grid_len=0.16,
        fine_grid_len=0.16,
        color_grid_len=0.16,
        hidden_size=32,
        pos_embedding_method="fourier",
        uncertainty="2D",
        num_sensors=1,
    ):
        super().__init__()

        self.uncertainty = uncertainty
        self.num_sensors = num_sensors

        self.middle_decoder = MLP(
            name="middle",
            dim=dim,
            c_dim=c_dim,
            color=False,
            skips=[2],
            n_blocks=5,
            hidden_size=hidden_size,
            grid_len=middle_grid_len,
            pos_embedding_method=pos_embedding_method,
        )
        self.fine_decoder = MLP(
            name="fine",
            dim=dim,
            c_dim=c_dim * 2,
            color=False,
            skips=[2],
            n_blocks=5,
            hidden_size=hidden_size,
            grid_len=fine_grid_len,
            concat_feature=True,
            pos_embedding_method=pos_embedding_method,
        )
        self.color_decoder = MLP(
            name="color",
            dim=dim,
            c_dim=c_dim,
            color=True,
            skips=[2],
            n_blocks=5,
            hidden_size=hidden_size,
            grid_len=color_grid_len,
            pos_embedding_method=pos_embedding_method,
        )
        self.color_conf_decoder = MLP_2D(name="c_var", dim=col_conf_dim, n_blocks=5, hidden_size=hidden_size)
        self.conf_decoders = []
        for i in range(num_sensors):
            self.conf_decoders.append(
                MLP_2D(
                    name=f"var{i}",
                    dim=conf_dim,
                    n_blocks=5,
                    hidden_size=hidden_size,
                )
            )
            self.conf_decoders[-1] = self.conf_decoders[-1].to("cuda:0")

    def forward(self, p, c_grid, stage="middle", d_feats=None, c_feats=None, **kwargs):
        """
        Output occupancy/color in different stage.
        """

        device = f"cuda:{p.get_device()}"

        if stage == "middle":
            middle_occ = self.middle_decoder(p, c_grid)
            middle_occ = middle_occ.squeeze(0)
            raw = torch.zeros(middle_occ.shape[0], 4).to(device).float()
            raw[..., -1] = middle_occ
        elif stage == "fine":
            fine_occ = self.fine_decoder(p, c_grid)
            raw = torch.zeros(fine_occ.shape[0], 4).to(device).float()
            middle_occ = self.middle_decoder(p, c_grid)
            middle_occ = middle_occ.squeeze(0)
            raw[..., -1] = fine_occ + middle_occ
        elif stage == "color":
            fine_occ = self.fine_decoder(p, c_grid)
            raw = self.color_decoder(p, c_grid)
            middle_occ = self.middle_decoder(p, c_grid)
            middle_occ = middle_occ.squeeze(0)
            raw[..., -1] = fine_occ + middle_occ

        if d_feats is not None:
            var = torch.zeros(d_feats.shape[0], self.num_sensors + 1).to(device).float()
            for i in range(self.num_sensors):
                var[..., i] = self.conf_decoders[i](d_feats[..., i]).squeeze()
            var[..., -1] = self.color_conf_decoder(c_feats).squeeze()
        else:
            var = torch.zeros(5000, self.num_sensors + 1).to(device).float()

        return (
            raw,
            var,
        )  # (N*samples,4) [color (3,), occupancy (1,)], (samples,2) [variance (1,)]
