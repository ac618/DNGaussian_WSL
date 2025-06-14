#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scipy.spatial.transform import Rotation as R, Slerp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
try:
    from scene.cameras import Camera
except:
    from scene_sh.cameras import Camera


depth_curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)

def weighted_percentile(x, w, ps, assume_sorted=False):
    """Compute the weighted percentile(s) of a single vector."""
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = np.argsort(x)
    x, w = x[sortidx], w[sortidx]
    acc_w = np.cumsum(w)
    return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)

def visualize_cmap(value, weight, colormap, lo=None, hi=None,
                   percentile=99., curve_fn=lambda x: x,
                   modulus=None, matte_background=True):
    """Visualize a 1D image and a 1D weighting according to some colormap.

    Args:
    value: A 1D image.
    weight: A weight map, in [0, 1].
    colormap: A colormap function.
    lo: The lower bound to use when rendering, if None then use a percentile.
    hi: The upper bound to use when rendering, if None then use a percentile.
    percentile: What percentile of the value map to crop to when automatically
      generating `lo` and `hi`. Depends on `weight` as well as `value'.
    curve_fn: A curve function that gets applied to `value`, `lo`, and `hi`
      before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
    modulus: If not None, mod the normalized value by `modulus`. Use (0, 1]. If
      `modulus` is not None, `lo`, `hi` and `percentile` will have no effect.
    matte_background: If True, matte the image over a checkerboard.

    Returns:
    A colormap rendering.
    """
    # Identify the values that bound the middle of `value' according to `weight`.
    lo_auto, hi_auto = weighted_percentile(
      value, weight, [50 - percentile / 2, 50 + percentile / 2])

    # If `lo` or `hi` are None, use the automatically-computed bounds above.
    eps = np.finfo(np.float32).eps
    lo = lo or (lo_auto - eps)
    hi = hi or (hi_auto + eps)

    # Curve all values.
    value, lo, hi = [curve_fn(x) for x in [value, lo, hi]]

    # Wrap the values around if requested.
    if modulus:
        value = np.mod(value, modulus) / modulus
    else:
        # Otherwise, just scale to [0, 1].
        value = np.nan_to_num(
        np.clip((value - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1))

    if colormap:
        colorized = colormap(value)[:, :, :3]
    else:
        assert len(value.shape) == 3 and value.shape[-1] == 3
        colorized = value

    return colorized

def plot_camera_frustum(ax, R, T, FoVx, FoVy, scale=20.0, color='r', label=''):
    """
    Plot a single camera frustum in 3D.
    R: 3x3 rotation matrix
    T: 3D translation vector
    FoVx, FoVy: field of view in degrees
    scale: depth scaling for visualization
    """
    # Create camera coordinate axes
    cam_dir = R.T @ np.array([0, 0, 1])
    cam_up = R.T @ np.array([0, -1, 0])
    cam_right = R.T @ np.array([1, 0, 0])

    # Calculate frustum corners
    depth = scale
    half_w = np.tan(np.radians(FoVx / 2)) * depth
    half_h = np.tan(np.radians(FoVy / 2)) * depth

    center = T
    forward = cam_dir * depth
    up = cam_up * half_h
    right = cam_right * half_w

    tl = center + forward - right + up
    tr = center + forward + right + up
    bl = center + forward - right - up
    br = center + forward + right - up

    # Draw pyramid lines
    corners = [tl, tr, br, bl]
    for corner in corners:
        ax.plot(*zip(center, corner), color=color)
    ax.plot(*zip(tl, tr, br, bl, tl), color=color)

    ax.scatter(*center, color=color, label=label)

def visualize_frustums(cam1, cam2, new_camera):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot_camera_frustum(ax, cam1.R, cam1.T, cam1.FoVx, cam1.FoVy, color='blue', label='cam1')
    plot_camera_frustum(ax, cam2.R, cam2.T, cam2.FoVx, cam2.FoVy, color='green', label='cam2')
    plot_camera_frustum(ax, new_camera.R, new_camera.T, new_camera.FoVx, new_camera.FoVy, color='red', label='interpolated')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_box_aspect([1,1,1])
    plt.title("Camera Frustum Visualization")
    plt.show()

def slerp_between_rotations(R1, R2, alpha):
    """
    Perform SLERP between two 3x3 rotation matrices.
    """
    key_times = [0, 1]
    key_rots = R.from_matrix([R1, R2])  # shape: (2, 3, 3)
    slerp = Slerp(key_times, key_rots)
    result_rot = slerp([alpha])  # Returns Rotation object at alpha
    return result_rot.as_matrix()[0]

def interpolate_cemeras(cam1, cam2, alpha, uid, args, method="slerp"):
    R1 = cam1.R
    R2 = cam2.R
    if method == "slerp":
        R_interp = slerp_between_rotations(R1, R2, alpha)
    elif method == "linear":
        R_interp = (1 - alpha) * R1 + alpha * R2
        U, _, Vt = np.linalg.svd(R_interp)
        R_interp = U @ Vt

    T_interp = (1 - alpha) * cam1.T + alpha * cam2.T

    FoVx = (1 - alpha) * cam1.FoVx + alpha * cam2.FoVx
    FoVy = (1 - alpha) * cam1.FoVy + alpha * cam2.FoVy

    image = cam1.original_image
    gt_alpha_mask = None
    depth_mono = None

    return Camera(
        colmap_id=uid,
        R=R_interp,
        T=T_interp,
        FoVx=FoVx,
        FoVy=FoVy,
        image=image,
        gt_alpha_mask=gt_alpha_mask,
        depth_mono=depth_mono,
        image_name=f"novel_view_{uid}",
        uid=uid,
        data_device=args.data_device
    )

def synthesize_novel_view(dataset, iteration, near, pipeline):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        (model_params, _) = torch.load(os.path.join(dataset.model_path, "chkpnt_latest.pth"))
        gaussians.restore(model_params)
        gaussians.neural_renderer.keep_sigma=True

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        orig_cameras = scene.getTrainCameras()
        cam1, cam2 = orig_cameras[1], orig_cameras[3]
        new_camera = interpolate_cemeras(
            cam1, cam2, 0.01, len(orig_cameras)+1, dataset, method="slerp"
        )
        # visualize_frustums(cam1, cam2, new_camera)
        views = [cam1, cam2, new_camera]

        render_path = os.path.join(dataset.model_path, 'novel', 'render')
        depth_path = os.path.join(dataset.model_path, 'novel', 'depth')
        makedirs(render_path, exist_ok=True)
        makedirs(depth_path, exist_ok=True)

        if near > 0:
            mask_near = None
            for idx, view in enumerate(tqdm(views, desc="Rendering progress", ascii=True, dynamic_ncols=True)):
                mask_temp = (gaussians.get_xyz - view.camera_center.repeat(gaussians.get_xyz.shape[0], 1)).norm(dim=1, keepdim=True) < near
                mask_near = mask_near + mask_temp if mask_near is not None else mask_temp
            gaussians.prune_points_inference(mask_near)
        
        for idx, view in enumerate(tqdm(views, desc="Rendering progress", ascii=True, dynamic_ncols=True)):
            render_pkg = render(view, gaussians, pipeline, background, inference=True)
            rendering = render_pkg["render"]
            depth = (render_pkg['depth'] - render_pkg['depth'].min()) / (render_pkg['depth'].max() - render_pkg['depth'].min()) + 1 * (1 - render_pkg["alpha"])
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

            depth_est = depth.squeeze().cpu().numpy()
            depth_est = visualize_cmap(depth_est, np.ones_like(depth_est), cm.get_cmap('turbo'), curve_fn=depth_curve_fn).copy()
            depth_est = torch.as_tensor(depth_est).permute(2,0,1)
            torchvision.utils.save_image(depth_est, os.path.join(depth_path, 'color_{0:05d}'.format(idx) + ".png"))
        

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--near", default=0, type=int)
    args = get_combined_args(parser)

    safe_state(args.quiet)

    synthesize_novel_view(model.extract(args), args.iteration, args.near, pipeline.extract(args))