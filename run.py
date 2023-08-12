import argparse
import os
import random

import numpy as np
import torch

from src import config
from src.uncle_slam import UncleSlam


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def main():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    setup_seed(20)

    parser = argparse.ArgumentParser(description="Arguments for running the UncLe-SLAM.")
    parser.add_argument("config", type=str, help="Path to config file.")
    parser.add_argument(
        "--input_folder",
        type=str,
        help="input folder, this have higher priority, can overwrite the one in config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="output folder, this have higher priority, can overwrite the one in config file",
    )
    uncle_parser = parser.add_mutually_exclusive_group(required=False)
    uncle_parser.add_argument("--uncle", dest="uncle", action="store_true")
    parser.set_defaults(uncle=True)
    args = parser.parse_args()

    cfg = config.load_config(args.config, "configs/uncle_slam.yaml")

    slam = UncleSlam(cfg, args)

    slam.run()


if __name__ == "__main__":
    main()
