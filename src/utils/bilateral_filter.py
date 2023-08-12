"""
Torch implementation of a bilateral filter for edge-preserving smoothing.

Adapted from:
https://gist.github.com/etienne87/9f903b2b16389f9fe98a18fade6df74b
"""

import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter


def generate_2d_kernel(length: int = 21, sigma: float = 3) -> torch.Tensor():
    """Generates a 2D Gaussian kernel array.

    Args:
        length: kernel dimensions
        sigma: standard deviation of kernel

    Returns:
        (length,length) 2D Gaussian kernel
    """
    a_range = torch.arange(-length // 2 + 1.0, length // 2 + 1.0)
    x_grid, y_grid = torch.meshgrid(a_range, a_range, indexing="xy")
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2.0 * sigma**2))
    return kernel


class Shift(nn.Module):
    """Module for generating colour-shifted tensor."""

    def __init__(self, channels: int, kernel_size: int = 3):
        """Module for  copying and shifting images for parallelization.

        Args:
            in_planes: # of input channels
            kernel_size: kernel dimension (square)
        """
        super(Shift, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def forward(self, x: torch.Tensor()):
        """Get expanded tensor for colour similarity.

        Args:
            x: (n,c,h,w) n images input with channels, height, width

        returns:
            (n, c*kernel_size**2, h, w) expanded tensor
        """
        _, c, h, w = x.size()

        # pad image based on kernel
        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
        cat_layers = []

        # cycle through channels, rows, columns
        for i in range(self.channels):
            # Parse in row-major
            for y in range(0, self.kernel_size):
                y2 = y + h
                for x in range(0, self.kernel_size):
                    x2 = x + w
                    xx = x_pad[:, i : i + 1, y:y2, x:x2]
                    cat_layers += [xx]

        return torch.cat(cat_layers, 1)


class BilateralFilter(nn.Module):
    """BilateralFilter computes:
    If = 1/W * Sum_{xi C Omega}(I * f(||I(xi)-I(x)||) * g(||xi-x||))
    """

    def __init__(
        self,
        channels: int,
        k: int,
        height: int,
        width: int,
        sigma_color: float = 10.0,
        sigma_space: float = 10.0,
    ):
        """Module for bilateral filter for edge-preserving smoothing.

        Args:
            channels: # channels in image
            k: kernel size
            height: height of image
            width: width of image
            sigma_color: standard deviation of colour smoothing
            sigma_space: standard deviation of space smoothing
        """
        super(BilateralFilter, self).__init__()

        # space gaussian kernel
        self.g = Parameter(torch.Tensor(channels, k * k))
        self.gw = gkern2d(k, sigma_space)
        self.g.data = torch.tile(self.gw.reshape(channels, k * k, 1, 1), (1, 1, height, width))  # (c,k*k,h,w)

        # get expanded shifted tensor
        self.shift = Shift(channels, k)
        self.sigma_color = 2 * sigma_color**2

    def forward(self, I: torch.Tensor()):
        """Get smoothed image.

        Args:
            I: image tensor (n,c,h,w)

        Returns:
            If: smoothed image (n,c,h,w)
        """
        I_shifted = self.shift(I).data
        I_expanded = I.expand(*I_shifted.size())

        # get weighted colour difference
        D = (I_shifted - I_expanded) ** 2  # here we are actually missing some sum over groups of channels
        De = torch.exp(-D / self.sigma_color)

        # multiply by weighted spacial difference
        Dd = De * self.g.data
        # print(Is.shape, De.shape, self.g.data.shape, Dd.shape)

        # get normalization constant
        W_denom = torch.sum(Dd, dim=1)

        # get filtered image by collapsing the expanded kernel dimension
        If = torch.sum(Dd * I_shifted, dim=1) / W_denom
        return If


if __name__ == "__main__":
    c, h, w = 1, 512, 512
    k = 7
    sigma_space, sigma_color = 10, 10

    device = "cuda" if torch.cuda.is_available() else "cpu"

    bilat = BilateralFilter(c, k, h, w, sigma_color, sigma_space)
    if device == "cuda":
        bilat.cuda()

    im = cv2.imread("depth000080.png", cv2.IMREAD_ANYDEPTH)
    im_in = im.reshape(1, 1, h, w)
    img = torch.from_numpy(im_in.astype(np.float64)).float()

    if device == "cuda":
        img_in = img.cuda()
    else:
        img_in = img

    start = time.time()
    y = bilat(img_in)
    print(time.time() - start)

    start = time.time()
    img_cv2 = cv2.bilateralFilter(im.astype(np.float32), k, sigma_color, sigma_space)
    print(time.time() - start)

    img_torch = y.cpu().numpy()[0]  # not counting the return transfer in timing!

    zy, zx = np.gradient(im)
    normal = np.dstack((-zx, -zy, np.ones_like(im)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n
    normal += 1
    normal /= 2
    normal[im == 0] = 0

    zy, zx = np.gradient(img_cv2)
    normal_cv2 = np.dstack((-zx, -zy, np.ones_like(im)))
    n = np.linalg.norm(normal_cv2, axis=2)
    normal_cv2[:, :, 0] /= n
    normal_cv2[:, :, 1] /= n
    normal_cv2[:, :, 2] /= n
    normal_cv2 += 1
    normal_cv2 /= 2
    normal_cv2[im == 0] = 0

    zy, zx = np.gradient(img_torch)
    normal_torch = np.dstack((-zx, -zy, np.ones_like(im)))
    n = np.linalg.norm(normal_torch, axis=2)
    normal_torch[:, :, 0] /= n
    normal_torch[:, :, 1] /= n
    normal_torch[:, :, 2] /= n
    normal_torch += 1
    normal_torch /= 2
    normal_torch[im == 0] = 0

    diff = np.abs(img_torch - img_cv2)
    diff_torch = np.abs(img_torch - im)
    diff_cv2 = np.abs(img_cv2 - im)
