import os
import cv2
import numpy as np
import argparse
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def parse_cameras_txt(path):
    intrinsics = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            elems = line.strip().split()
            cam_id = int(elems[0])
            model = elems[1]
            width, height = map(int, elems[2:4])
            params = list(map(float, elems[4:]))
            intrinsics[cam_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params
            }
    return intrinsics

def parse_images_txt(path):
    cameras = {}
    processed_lines = []
    with open(path, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#") or line.strip() == "":
            i += 1
            continue
        processed_lines.append(line)
        elems = line.strip().split()
        image_id = int(elems[0])
        qvec = np.array(list(map(float, elems[1:5])))
        tvec = np.array(list(map(float, elems[5:8])))
        cam_id = int(elems[8])
        name = elems[9]
        cameras[image_id] = {
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": cam_id,
            "name": name
        }
        i += 2  # Skip POINT2D[] line
    return cameras, processed_lines

def qvec2rotmat(qvec):
    return R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()

def backproject(depth, intrinsics, img_mask, depth_thresh=None, scale=1.0):
    h, w = depth.shape
    fx, fy, cx, cy = intrinsics[:4]
    fx *= scale
    fy *= scale
    cx *= scale
    cy *= scale

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.astype(np.float32).flatten()
    y = y.astype(np.float32).flatten()
    z = depth.flatten()

    valid_mask = (z > 0)
    if depth_thresh is not None:
        valid_mask &= (z < depth_thresh)

    img_valid = np.zeros_like(depth, dtype=bool)
    img_valid[img_mask[0]:img_mask[1]+1, img_mask[2]:img_mask[3]+1] = True
    img_valid = img_valid.flatten()
    valid_mask &= img_valid

    x = x[valid_mask]
    y = y[valid_mask]
    z = z[valid_mask]

    x = (x - cx) / fx
    y = (y - cy) / fy
    X = np.vstack((x * z, y * z, z)).T
    return X, valid_mask

def transform_points(X_cam, R, t):
    return (R.T @ (X_cam - t).T).T

def project_points(X_world, R, t, intrinsics):
    P = (R @ X_world.T + t.reshape(3,1)).T
    fx, fy, cx, cy = intrinsics[:4]
    x = (P[:, 0] / P[:, 2]) * fx + cx
    y = (P[:, 1] / P[:, 2]) * fy + cy
    valid = (P[:, 2] > 0)
    return x, y, valid

def mask_valid_image(img):
    mask = np.any(img != [0, 0, 0], axis=-1)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    mask = [rmin, rmax, cmin, cmax]
    return mask

def visualize_point_cloud(points, sample_rate=1, colors=None):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    sampled_points = points[::sample_rate]
    
    if colors is not None:
        # Ensure proper float conversion and normalization
        sampled_colors = colors[::sample_rate].astype(np.float32) / 255.0
    else:
        sampled_colors = 'b'

    xs = sampled_points[:, 0]
    ys = sampled_points[:, 1]
    zs = sampled_points[:, 2]

    ax.scatter(xs, ys, zs, s=0.5, c=sampled_colors)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=30, azim=-90)
    plt.tight_layout()
    plt.show()

def visualize_depth_map(depth, img_mask, depth_thresh=None, save_path=None):
    mask = np.zeros_like(depth, dtype=bool)
    mask[img_mask[0]:img_mask[1]+1, img_mask[2]:img_mask[3]+1] = True
    mask &= (depth > 0)
    if depth_thresh is not None:
        mask &= (depth < depth_thresh)

    masked_depth = np.where(mask, depth, np.nan)
    plt.figure(figsize=(8, 6))
    plt.imshow(masked_depth, cmap='viridis')
    plt.colorbar(label='Depth')
    plt.title("Masked Depth Map")
    plt.axis('off')
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def save_image_points_original(img, save_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Image Points (Original Color, No Mask)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def save_image_points_masked(img, img_mask, save_path):
    masked_img = img.copy()
    rmin, rmax, cmin, cmax = img_mask
    masked_img[:rmin, :] = 0
    masked_img[rmax+1:, :] = 0
    masked_img[:, :cmin] = 0
    masked_img[:, cmax+1:] = 0

    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
    plt.title("Image Points (Original Color, With Mask)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
    cam_depths = [(30.0, 60.0), (30.0, 60.0), (30.0, 60.0), (30.0, 60.0)]

    parser = argparse.ArgumentParser(description="Get Points2D of COLMAP")
    parser.add_argument("--data_dir", required=True, type=str)
    args = parser.parse_args()

    data_dir = args.data_dir
    image_dir = os.path.join(data_dir, 'images')
    depth_dir = os.path.join(data_dir, 'depth_maps')
    sparse_dir = os.path.join(data_dir, 'sparse/0') 
    camera_file = os.path.join(sparse_dir, 'cameras.txt')
    image_file = os.path.join(sparse_dir, 'images.txt')
    output_image_file = os.path.join(sparse_dir, 'new_images.txt')
    output_point_file = os.path.join(sparse_dir, 'points3D.txt')
    # plot_dir = os.path.join(data_dir, 'visualizations')
    # os.makedirs(plot_dir, exist_ok=True)

    intrinsics_dict = parse_cameras_txt(camera_file)
    cameras, image_lines = parse_images_txt(image_file)
    all_points = []
    points2D_map = defaultdict(list)
    point_id = 1

    for image_id, cam in cameras.items():
        cam_id = cam["camera_id"]
        cam_info = intrinsics_dict[cam_id]
        intr = cam_info["params"]
        full_h, full_w = cam_info["height"], cam_info["width"]
        qvec = cam["qvec"]
        tvec = cam["tvec"]
        Rmat = qvec2rotmat(qvec)
        t = np.array(tvec)

        img_name = cam['name']
        img_path = os.path.join(image_dir, img_name)
        depth_path = os.path.join(depth_dir, f"depth_{img_name.split('.')[0]}.png")
        # base_name = img_name.split('/')[0]

        rgb = cv2.imread(str(img_path))
        img_mask = mask_valid_image(rgb)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
        depth = cv2.resize(depth, (full_w, full_h), interpolation=cv2.INTER_NEAREST)

        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32)

        depth = cam_depths[cam_id - 1][1] - depth * (cam_depths[cam_id - 1][1] - cam_depths[cam_id - 1][0]) / 255.0

        # Save visualizations
        # save_image_points_original(rgb, os.path.join(plot_dir, f"{base_name}_orig_color.png"))
        # save_image_points_masked(rgb, img_mask, os.path.join(plot_dir, f"{base_name}_masked_color.png"))
        # visualize_depth_map(depth, img_mask, depth_thresh=cam_depths[cam_id - 1][1] / 2,
        #                     save_path=os.path.join(plot_dir, f"{base_name}_depth_masked.png"))

        points_cam, valid_mask = backproject(depth, intr, img_mask, depth_thresh=45, scale=1.0)
        rgb_flat = rgb.reshape(-1, 3)
        valid_colors = rgb_flat[valid_mask]
        # print(valid_colors.shape)
        # save_image_points_original(valid_colors, os.path.join(plot_dir, f"{base_name}_masked_color.png"))
        points_world = transform_points(points_cam, Rmat, t)
        # visualize_point_cloud(points_world, sample_rate=1, colors=valid_colors)
        # visualize_point_cloud(points_world, sample_rate=1, colors=valid_colors)
        
        for i in range(0, points_world.shape[0], 400):
            X = points_world[i]
            color = valid_colors[i]
            all_points.append((point_id, X, color))
            x_proj, y_proj, valid = project_points(np.array([X]), Rmat, t, intr)
            if valid[0]:
                points2D_map[image_id].append((x_proj[0], y_proj[0], point_id))
            point_id += 1

    with open(output_point_file, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: {}\n".format(len(all_points)))

        for pid, X, color in all_points:
            R, G, B = color
            f.write(f"{pid} {X[0]} {X[1]} {X[2]} {int(R)} {int(G)} {int(B)} 1.0 \n")

    with open(output_image_file, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        i = 0
        while i < len(image_lines):
            line = image_lines[i]
            if line.startswith("#") or line.strip() == "":
                f.write(line)
                i += 1
                continue
            f.write(line)
            # i += 1
            image_id = int(line.strip().split()[0])
            pts = points2D_map[image_id]
            point_strs = [f"{x:.2f} {y:.2f} {pid}" for x, y, pid in pts]
            f.write(" ".join(point_strs) + "\n")
            i += 1

if __name__ == "__main__":
    main()