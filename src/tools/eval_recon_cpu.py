import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree as KDTree

sys.path.append(".")
from src import config


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def completion_ratio(gt_points, rec_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(np.float64))
    return comp_ratio


def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc = np.mean(distances)
    return acc


def completion(gt_points, rec_points):
    gt_points_kd_tree = KDTree(rec_points)
    distances, _ = gt_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    return comp


def get_align_transformation(rec_meshfile, gt_meshfile):
    """
    Get the transformation matrix to align the reconstructed mesh to the ground truth mesh.
    """
    o3d_rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    o3d_gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    o3d_rec_pc = o3d.geometry.PointCloud(points=o3d_rec_mesh.vertices)
    o3d_gt_pc = o3d.geometry.PointCloud(points=o3d_gt_mesh.vertices)
    trans_init = np.eye(4)
    threshold = 0.1
    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d_rec_pc,
        o3d_gt_pc,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    transformation = reg_p2p.transformation
    return transformation


def check_proj(points, W, H, fx, fy, cx, cy, c2w):
    """
    Check if points can be projected into the camera view.

    """
    c2w = c2w.copy()
    c2w[:3, 1] *= -1.0
    c2w[:3, 2] *= -1.0

    w2c = np.linalg.inv(c2w)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    ones = np.ones_like(points[:, 0]).reshape(-1, 1)
    homo_points = np.concatenate([points, ones], axis=1).reshape(-1, 4, 1)  # (N, 4)
    cam_cord_homo = w2c @ homo_points  # (N, 4, 1)=(4,4)*(N, 4, 1)
    cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)
    cam_cord[:, 0] *= -1
    uv = K @ cam_cord
    z = uv[:, -1:] + 1e-5
    uv = uv[:, :2] / z
    uv = uv.squeeze(-1)
    edge = 0
    mask = (0 <= -z[:, 0, 0]) & (uv[:, 0] < W - edge) & (uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)

    return mask.sum() > 0


def calc_3d_metric(rec_meshfile, gt_meshfile, align=True):
    """
    3D reconstruction metric.

    """
    mesh_rec = trimesh.load(rec_meshfile, process=False)
    mesh_gt = trimesh.load(gt_meshfile, process=False)

    if align:
        transformation = get_align_transformation(rec_meshfile, gt_meshfile)
        mesh_rec = mesh_rec.apply_transform(transformation)

    rec_pc = trimesh.sample.sample_surface(mesh_rec, 200000)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, 200000)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])
    accuracy_rec = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_rec = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_ratio_rec = completion_ratio(gt_pc_tri.vertices, rec_pc_tri.vertices)
    accuracy_rec *= 100  # convert to cm
    completion_rec *= 100  # convert to cm
    completion_ratio_rec *= 100  # convert to %
    print("accuracy: ", accuracy_rec)
    print("completion: ", completion_rec)
    print("completion ratio: ", completion_ratio_rec)
    return accuracy_rec, completion_rec, completion_ratio_rec


def get_cam_position(gt_meshfile):
    mesh_gt = trimesh.load(gt_meshfile)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh_gt)
    extents[2] *= 0.7
    extents[1] *= 0.7
    extents[0] *= 0.3
    transform = np.linalg.inv(to_origin)
    transform[2, 3] += 0.4
    return extents, transform


def calc_2d_metric(rec_meshfile, gt_meshfile, align=True, n_imgs=1000):
    """
    2D reconstruction metric, depth L1 loss.

    """
    H = 500
    W = 500
    focal = 300
    fx = focal
    fy = focal
    cx = H / 2.0 - 0.5
    cy = W / 2.0 - 0.5

    gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    unseen_gt_pointcloud_file = gt_meshfile.replace(".ply", "_pc_unseen.npy")
    pc_unseen = np.load(unseen_gt_pointcloud_file)
    if align:
        transformation = get_align_transformation(rec_meshfile, gt_meshfile)
        rec_mesh = rec_mesh.transform(transformation)

    # get vacant area inside the room
    extents, transform = get_cam_position(gt_meshfile)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W, height=H)
    vis.get_render_option().mesh_show_back_face = True
    errors = []
    for i in range(n_imgs):
        while True:
            # sample view, and check if unseen region is not inside the camera view
            # if inside, then needs to resample
            up = [0, 0, -1]
            origin = trimesh.sample.volume_rectangular(extents, 1, transform=transform)
            origin = origin.reshape(-1)
            tx = round(random.uniform(-10000, +10000), 2)
            ty = round(random.uniform(-10000, +10000), 2)
            tz = round(random.uniform(-10000, +10000), 2)
            target = [tx, ty, tz]
            target = np.array(target) - np.array(origin)
            c2w = viewmatrix(target, up, origin)
            tmp = np.eye(4)
            tmp[:3, :] = c2w
            c2w = tmp
            seen = check_proj(pc_unseen, W, H, fx, fy, cx, cy, c2w)
            if ~seen:
                break

        param = o3d.camera.PinholeCameraParameters()
        param.extrinsic = np.linalg.inv(c2w)  # 4x4 numpy array

        param.intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

        ctr = vis.get_view_control()
        ctr.set_constant_z_far(20)
        ctr.convert_from_pinhole_camera_parameters(param)

        vis.add_geometry(
            gt_mesh,
            reset_bounding_box=True,
        )
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()
        gt_depth = vis.capture_depth_float_buffer(True)
        gt_depth = np.asarray(gt_depth)
        vis.remove_geometry(
            gt_mesh,
            reset_bounding_box=True,
        )

        vis.add_geometry(
            rec_mesh,
            reset_bounding_box=True,
        )
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()
        ours_depth = vis.capture_depth_float_buffer(True)
        ours_depth = np.asarray(ours_depth)
        vis.remove_geometry(
            rec_mesh,
            reset_bounding_box=True,
        )

        errors += [np.abs(gt_depth - ours_depth).mean()]

    errors = np.array(errors)
    # from m to cm
    print("Depth L1: ", errors.mean() * 100)
    return errors.mean() * 100


if __name__ == "__main__":
    np.random.seed(142857)
    random.seed(285714)

    parser = argparse.ArgumentParser(description="Arguments to evaluate the reconstruction.")
    parser.add_argument("config", type=str, help="Path to config file.")
    parser.add_argument("--rec_mesh", type=str, help="reconstructed mesh file path")
    parser.add_argument("--gt_mesh", type=str, help="ground truth mesh file path")
    parser.add_argument("-2d", "--metric_2d", action="store_true", help="enable 2D metric")
    parser.add_argument("-3d", "--metric_3d", action="store_true", help="enable 3D metric")
    args = parser.parse_args()

    cfg = config.load_config(args.config, "configs/uncle_slam.yaml")

    if args.metric_3d:
        acc, comp, comp_r = calc_3d_metric(args.rec_mesh, args.gt_mesh)

    err2 = 0

    if args.metric_2d:
        err2 = calc_2d_metric(args.rec_mesh, args.gt_mesh, n_imgs=1000)

    eval_text_path = Path("eval/")
    if not eval_text_path.exists():
        eval_text_path.mkdir()

    recon_mesh = args.rec_mesh.split("/")

    config_path = args.config.split("/")

    scene = config_path[-1].rstrip(".yaml")

    with open(eval_text_path / "recon_results.csv", "a", newline="") as csvfile:
        recon_writer = csv.writer(csvfile, delimiter=",")
        recon_writer.writerow([f"{scene}", acc, comp, comp_r, err2])
