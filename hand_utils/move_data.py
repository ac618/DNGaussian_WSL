import os
import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

def write_camera_txt(camera_file, intrinsics_dict, width, height):
    with open(camera_file, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: {}\n".format(len(intrinsics_dict)))

        for i, K in intrinsics_dict.items():
            fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
            f.write(f"{int(i)+1} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

def write_image_txt(image_file, image_paths, extrinsics_dict, time_id):
    sub_dirs = sorted(os.listdir(image_paths))
    with open(image_file, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        image_id = 1
        for i, w2c in extrinsics_dict.items():
            w2c = np.array(w2c)
            R_wc, t_wc = w2c[:3, :3], w2c[:3, 3]
            R_cw = R_wc.T
            t_cw = -R_cw @ t_wc
            quat = R.from_matrix(R_cw).as_quat()
            qx, qy, qz, qw = quat
            tx, ty, tz = t_cw

            imgs = sorted(os.listdir(os.path.join(image_paths, sub_dirs[int(i)])))
            for j, img_name in enumerate(imgs):
                if (j == time_id):
                    f.write(f"{image_id} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {int(i)+1} {sub_dirs[int(i)]}.jpg\n")
                    f.write(f"\n")
                    image_id += 1

def move_images(image_paths, output_path, time_id):
    sub_dirs = sorted(os.listdir(image_paths))
    for i in range(len(sub_dirs)):
        imgs = sorted(os.listdir(os.path.join(image_paths, sub_dirs[i])))
        for j, img_name in enumerate(imgs):
            if (j == time_id):
                os.system("cp {target} {output}".format(
                    target=os.path.join(image_paths, sub_dirs[i], img_name),
                    output=os.path.join(output_path, f'{int(sub_dirs[i])}.jpg')
                ))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data Preparation (COLMAP Format)")
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--output_path', required=True, type=str)
    parser.add_argument('--time_id', required=True, type=int)
    
    args = parser.parse_args()
    imgs_path = os.path.join(args.data_path, 'frames_with_metadata/normal/ims')
    intrinsic_path = os.path.join(args.data_path, 'intrinsic.json')
    extrinsic_path = os.path.join(args.data_path, 'extrinsic.json')
    info_path = os.path.join(args.data_path, 'frames_with_metadata/normal/train_meta.json')

    with open(info_path, 'rb') as info_file:
        info_data = json.load(info_file)
    img_width = info_data['w']
    img_height = info_data['h']

    with open(intrinsic_path, 'rb') as int_file:
        k_matrices = json.load(int_file)

    with open(extrinsic_path, 'rb') as ext_file:
        w2c_matrices = json.load(ext_file)

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, str(args.time_id)), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, str(args.time_id), 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, str(args.time_id), 'sparse'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, str(args.time_id), 'sparse', '0'), exist_ok=True)

    ## Write cameras.txt
    camera_file = os.path.join(args.output_path, str(args.time_id), 'sparse', '0', 'cameras.txt')
    write_camera_txt(camera_file, k_matrices, img_width, img_height)

    ## Write images.txt
    image_file = os.path.join(args.output_path, str(args.time_id), 'sparse', '0', 'images.txt')
    write_image_txt(image_file, imgs_path, w2c_matrices, args.time_id)

    ## Move images
    output_imgs_path = os.path.join(args.output_path, str(args.time_id), 'images')
    move_images(imgs_path, output_imgs_path, args.time_id)