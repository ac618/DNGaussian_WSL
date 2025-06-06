import os
import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

def project_point(K, w2c, point3D):
    """Projects a 3D point into 2D using K and w2c matrix."""
    point_h = np.append(point3D[:3], 1.0)  # Homogeneous
    cam_point = w2c @ point_h  # 3x4 @ 4x1
    if cam_point[2] <= 0:
        return None
    pixel = K @ cam_point[:3]
    pixel = pixel[:2] / cam_point[2]
    return pixel

def write_camera_txt(camera_file, intrinsics_list, width, height):
    with open(camera_file, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: {}\n".format(len(intrinsics_list)))

        for i, K in enumerate(intrinsics_list):
            fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
            f.write(f"{i+1} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

def write_image_txt(image_file, image_paths, extrinsics_list):
    sub_dirs = sorted(os.listdir(image_paths))
    with open(image_file, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        image_id = 1
        for i, w2c in enumerate(extrinsics_list):
            w2c = np.array(w2c)
            R_wc, t_wc = w2c[:3, :3], w2c[:3, 3]
            R_cw = R_wc.T
            t_cw = -R_cw @ t_wc
            quat = R.from_matrix(R_cw).as_quat()
            qx, qy, qz, qw = quat
            tx, ty, tz = t_cw

            imgs = sorted(os.listdir(os.path.join(image_paths, sub_dirs[i])))
            for j, img_name in enumerate([imgs[0]]):
                f.write(f"{image_id} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {i+1} {os.path.join(sub_dirs[i], img_name)}\n")
                f.write(f"\n")
                image_id += 1

def write_point_txt(point_file, npz_path):
    data = np.load(npz_path)['data']
    with open(point_file, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: {}\n".format(data.shape[0]))

        for i in range(data.shape[0]):
            point = data[i]
            f.write(f"{i+1} {point[0]} {point[1]} {point[2]} {int(point[3]*255)} {int(point[4]*255)} {int(point[5]*255)} {0.0}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data Preparation (COLMAP Format)")
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--output_path', required=True, type=str)
    parser.add_argument('--mask_path', default="../data/masks", type=str)

    args = parser.parse_args()
    imgs_path = os.path.join(args.data_path, 'ims')
    mask_path = os.path.join(args.data_path, 'seg')
    npz_path = os.path.join(args.data_path, 'init_pt_cld.npz')
    info_path = os.path.join(args.data_path, 'train_meta.json')

    with open(info_path, 'rb') as info_file:
        info_data = json.load(info_file)
    img_width = info_data['w']
    img_height = info_data['h']
    k_matrices = info_data['k']
    w2c_matrices = info_data['w2c']

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'sparse'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'sparse', '0'), exist_ok=True)

    ## Write camera.txt
    camera_file = os.path.join(args.output_path, 'sparse', '0', 'cameras.txt')
    write_camera_txt(camera_file, k_matrices[0], img_width, img_height)

    ## Write images.txt
    image_file = os.path.join(args.output_path, 'sparse', '0', 'images.txt')
    write_image_txt(image_file, imgs_path, w2c_matrices[0])

    """
    ## Write points3D.txt
    point_file = os.path.join(args.output_path, 'sparse', '0', 'points3D.txt')
    write_point_txt(point_file, npz_path)
    """

    ## Move images
    os.system("cp -r {target} {output}".format(
        target=imgs_path,
        output=os.path.join(args.output_path, 'images')
    ))

    ## Move masks
    os.system("cp -r {target} {output}".format(
        target=mask_path,
        output=args.mask_path
    ))