import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from evaluate_3d_reconstruction import run_evaluation

sys.path.append(".")
from src import config

if __name__ == "__main__":
    np.random.seed(142857)

    parser = argparse.ArgumentParser(description="Arguments to evaluate the reconstruction.")
    parser.add_argument("config", type=str, help="Path to config file.")
    parser.add_argument("--rec_path", type=str, help="reconstructed mesh file path")
    args = parser.parse_args()

    cfg = config.load_config(args.config, "configs/uncle_slam.yaml")

    eval_text_path = Path("eval/")
    if not eval_text_path.exists():
        eval_text_path.mkdir()

    config_path = args.config.split("/")

    # scene = config_path[-1].rstrip('.yaml')
    scene = cfg["data"]["input_folder"].split("/")[-2]

    pr, rc, f1, mean_pr, mean_rc = run_evaluation("final_mesh_eval_rec.ply", args.rec_path, scene, distance_thresh=0.05)

    with open(eval_text_path / "f1_results.csv", "a", newline="") as csvfile:
        recon_writer = csv.writer(csvfile, delimiter=",")
        recon_writer.writerow([f"{scene}", pr, rc, f1, mean_pr, mean_rc])
