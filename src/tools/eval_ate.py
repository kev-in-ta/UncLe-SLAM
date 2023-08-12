import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(".")
from src import config
from src.common import get_tensor_from_camera


def associate(first_list, second_list, offset=0.0, max_difference=0.02):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
    to find the closest match for every input tuple.

    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

    """
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [
        (abs(a - (b + offset)), a, b) for a in first_keys for b in second_keys if abs(a - (b + offset)) < max_difference
    ]
    potential_matches.sort()
    matches = []
    for _, first, second in potential_matches:
        if first in first_keys and second in second_keys:
            first_keys.remove(first)
            second_keys.remove(second)
            matches.append((first, second))

    matches.sort()
    return matches


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)

    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)

    """
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(axis=1, keepdims=True)
    data_zerocentered = data - data.mean(axis=1, keepdims=True)

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, _, Vh = np.linalg.linalg.svd(W.T)
    S = np.identity(3)
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1
    rot = U @ S @ Vh
    trans = data.mean(axis=1, keepdims=True) - rot @ model.mean(axis=1, keepdims=True)

    model_aligned = rot @ model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0))

    return rot, trans, trans_error


def plot_traj(ax, stamps, traj, style, color, label):
    """
    Plot a trajectory using matplotlib.

    Input:
    ax -- the plot
    stamps -- time stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend

    """
    stamps.sort()
    interval = np.median([s - t for s, t in zip(stamps[1:], stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i] - last < 2 * interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
        elif len(x) > 0:
            ax.plot(x, y, style, color=color, label=label)
            label = ""
            x = []
            y = []
        last = stamps[i]
    if len(x) > 0:
        ax.plot(x, y, style, color=color, label=label)


def evaluate_ate(first_list, second_list, plot="", _args=""):
    """Compute the absolute trajectory error."""
    # parse command line
    parser = argparse.ArgumentParser(
        description="This script computes the absolute trajectory error from the ground truth trajectory and the estimated trajectory."
    )
    # parser.add_argument('first_file', help='ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)')
    # parser.add_argument('second_file', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument(
        "--offset",
        help="time offset added to the timestamps of the second file (default: 0.0)",
        default=0.0,
    )
    parser.add_argument(
        "--scale",
        help="scaling factor for the second trajectory (default: 1.0)",
        default=1.0,
    )
    parser.add_argument(
        "--max_difference",
        help="maximally allowed time difference for matching entries (default: 0.02)",
        default=0.02,
    )
    parser.add_argument(
        "--save",
        help="save aligned second trajectory to disk (format: stamp2 x2 y2 z2)",
    )
    parser.add_argument(
        "--save_associations",
        help="save associated first and aligned second trajectory to disk (format: stamp1 x1 y1 z1 stamp2 x2 y2 z2)",
    )
    parser.add_argument(
        "--plot",
        help="plot the first and the aligned second trajectory to an image (format: png)",
    )
    parser.add_argument(
        "--verbose",
        help="print all evaluation data (otherwise, only the RMSE absolute translational error in meters after alignment will be printed)",
        action="store_true",
    )
    args = parser.parse_args(_args)
    args.plot = plot
    # first_list = associate.read_file_list(args.first_file)
    # second_list = associate.read_file_list(args.second_file)

    matches = associate(first_list, second_list, float(args.offset), float(args.max_difference))
    if len(matches) < 2:
        raise ValueError(
            "Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! \
            Did you choose the correct sequence?"
        )

    first_xyz = np.asarray([[float(value) for value in first_list[a][0:3]] for a, b in matches]).T
    second_xyz = np.asarray([[float(value) * float(args.scale) for value in second_list[b][0:3]] for a, b in matches]).T

    rot, trans, trans_error = align(second_xyz, first_xyz)

    second_xyz_aligned = rot @ second_xyz + trans

    first_stamps = list(first_list.keys())
    first_stamps.sort()
    first_xyz_full = np.asarray([[float(value) for value in first_list[b][0:3]] for b in first_stamps]).T

    second_stamps = list(second_list.keys())
    second_stamps.sort()
    second_xyz_full = np.asarray(
        [[float(value) * float(args.scale) for value in second_list[b][0:3]] for b in second_stamps]
    ).T

    second_xyz_full_aligned = rot @ second_xyz_full + trans

    if args.verbose:
        print("compared_pose_pairs %d pairs" % (len(trans_error)))

        print("absolute_translational_error.rmse %f m" % np.sqrt(np.dot(trans_error, trans_error) / len(trans_error)))
        print("absolute_translational_error.mean %f m" % np.mean(trans_error))
        print("absolute_translational_error.median %f m" % np.median(trans_error))
        print("absolute_translational_error.std %f m" % np.std(trans_error))
        print("absolute_translational_error.min %f m" % np.min(trans_error))
        print("absolute_translational_error.max %f m" % np.max(trans_error))

    if args.save_associations:
        file = open(args.save_associations, "w")
        file.write(
            "\n".join(
                [
                    "%f %f %f %f %f %f %f %f" % (a, x1, y1, z1, b, x2, y2, z2)
                    for (a, b), (x1, y1, z1), (x2, y2, z2) in zip(
                        matches,
                        first_xyz.T,
                        second_xyz_aligned.T,
                    )
                ]
            )
        )
        file.close()

    if args.save:
        with open(args.save, "w") as file:
            file.write(
                "\n".join(
                    [
                        "%f " % stamp + " ".join(["%f" % d for d in line])
                        for stamp, line in zip(second_stamps, second_xyz_full_aligned.T)
                    ]
                )
            )

    if args.plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ATE = np.sqrt(np.dot(trans_error, trans_error) / trans_error.shape)
        ax.set_title(f"len:{len(trans_error)} ATE RMSE:{ATE} {args.plot[:-3]}")
        plot_traj(ax, first_stamps, first_xyz_full.T, "-", "black", "ground truth")
        for i in range(first_xyz_full.T.shape[0]):
            gt2est = [
                [first_xyz_full.T[i][0], second_xyz_full_aligned.T[i][0]],
                [first_xyz_full.T[i][1], second_xyz_full_aligned.T[i][1]],
            ]
            plt.plot(gt2est[0], gt2est[1], color="gray", alpha=1.0, lw=0.5)
        plot_traj(
            ax,
            second_stamps,
            second_xyz_full_aligned.T,
            "-",
            "blue",
            "estimated",
        )
        ax.plot(
            first_xyz_full.T[0][0],
            first_xyz_full.T[0][1],
            marker="o",
            markersize=10,
            markerfacecolor="green",
            label="start gt",
        )
        ax.plot(
            first_xyz_full.T[-1][0],
            first_xyz_full.T[-1][1],
            marker="o",
            markersize=10,
            markerfacecolor="yellow",
            label="end gt",
        )
        ax.plot(
            second_xyz_full_aligned.T[0][0],
            second_xyz_full_aligned.T[0][1],
            marker="o",
            markersize=5,
            markerfacecolor="cyan",
            label="start estimated",
        )
        ax.plot(
            second_xyz_full_aligned.T[-1][0],
            second_xyz_full_aligned.T[-1][1],
            marker="o",
            markersize=5,
            markerfacecolor="purple",
            label="end estimated",
        )

        label = "difference"
        for (a, b), (x1, y1, z1), (x2, y2, z2) in zip(matches, first_xyz.T, second_xyz_aligned.T):
            # ax.plot([x1,x2],[y1,y2],'-',color="red",label=label)
            label = ""
        ax.legend()
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        plt.savefig(args.plot, dpi=90)

    return {
        "compared_pose_pairs": (len(trans_error)),
        "absolute_translational_error.rmse": np.sqrt(np.dot(trans_error, trans_error) / len(trans_error)),
        "absolute_translational_error.mean": np.mean(trans_error),
        "absolute_translational_error.median": np.median(trans_error),
        "absolute_translational_error.std": np.std(trans_error),
        "absolute_translational_error.min": np.min(trans_error),
        "absolute_translational_error.max": np.max(trans_error),
    }


def evaluate(poses_gt, poses_est, plot):
    poses_gt = poses_gt.cpu().numpy()
    poses_est = poses_est.cpu().numpy()

    N = poses_gt.shape[0]
    poses_gt = dict([(i, poses_gt[i]) for i in range(N)])
    poses_est = dict([(i, poses_est[i]) for i in range(N)])

    results = evaluate_ate(poses_gt, poses_est, plot)
    print(results)

    return results


def convert_poses(c2w_list, N, scale, gt=True):
    poses = []
    mask = torch.ones(N + 1).bool()
    for idx in range(0, N + 1):
        if gt:
            # some frame have `nan` or `inf` in gt pose of ScanNet,
            # but our system have estimated camera pose for all frames
            # therefore, when calculating the pose error, we need to mask out invalid pose
            if torch.isinf(c2w_list[idx]).any():
                mask[idx] = 0
                continue
            if torch.isnan(c2w_list[idx]).any():
                mask[idx] = 0
                continue
        c2w_list[idx][:3, 3] /= scale
        poses.append(get_tensor_from_camera(c2w_list[idx], Tquad=True))
    poses = torch.stack(poses)
    return poses, mask


if __name__ == "__main__":
    """This ATE evaluation code is modified upon the evaluation code in lie-torch."""

    parser = argparse.ArgumentParser(description="Arguments to eval the tracking ATE.")
    parser.add_argument("config", type=str, help="Path to config file.")
    parser.add_argument(
        "--output",
        type=str,
        help="output folder, this have higher priority, can overwrite the one inconfig file",
    )
    uncle_parser = parser.add_mutually_exclusive_group(required=False)
    uncle_parser.add_argument("--uncle", dest="uncle", action="store_true")
    parser.set_defaults(uncle=True)

    args = parser.parse_args()
    cfg = config.load_config(args.config, "configs/uncle_slam.yaml")
    scale = cfg["scale"]
    output = cfg["data"]["output"] if args.output is None else args.output
    ckptsdir = f"{output}/ckpts"
    if os.path.exists(ckptsdir):
        ckpts = [os.path.join(ckptsdir, f) for f in sorted(os.listdir(ckptsdir)) if "tar" in f]
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print("Get ckpt :", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
            estimate_c2w_list = ckpt["estimate_c2w_list"]
            gt_c2w_list = ckpt["gt_c2w_list"]
            N = ckpt["idx"]
            poses_gt, mask = convert_poses(gt_c2w_list, N, scale)
            poses_est, _ = convert_poses(estimate_c2w_list, N, scale)
            poses_est = poses_est[mask]
            result = evaluate(poses_gt, poses_est, plot=f"{output}/eval_ate_plot.png")

            eval_text_path = Path("eval/")
            if not eval_text_path.exists():
                eval_text_path.mkdir()

            config_path = args.config.split("/")

            scene = config_path[-1].rstrip(".yaml")

            with open(eval_text_path / "ate_results.csv", "a", newline="") as csvfile:
                recon_writer = csv.writer(csvfile, delimiter=",")
                recon_writer.writerow([f"{scene}", result["absolute_translational_error.rmse"]])
