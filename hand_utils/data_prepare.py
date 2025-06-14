import os
import argparse

parser = argparse.ArgumentParser(description="Prepare Hand Data")
parser.add_argument("--data_dir", required=True, type=str)
parser.add_argument("--output_dir", required=True, type=str)

args = parser.parse_args()
num_imgs = len(os.listdir(os.path.join(args.data_dir, 'frames_with_metadata/normal/ims/0')))
for i in range(num_imgs):
    time_id = i
    os.system("python move_data.py --data_path {data_path} --output_path {output_path} --time_id {time_id}".format(
        data_path=args.data_dir,
        output_path=args.output_dir,
        time_id=time_id
    ))
    os.system("python ../dpt/get_depth_map_for_llff_dtu.py --root_path {root_path} --scenes images".format(
        root_path=os.path.join(args.output_dir, str(time_id))
    ))
    os.system("mv {target} {output}".format(
        target=os.path.join(args.output_dir, str(time_id), 'images', 'depth_maps'),
        output=os.path.join(args.output_dir, str(time_id))
    ))
    os.system("python get_point2d.py --data_dir {data_dir}".format(
        data_dir=os.path.join(args.output_dir, str(time_id))
    ))
    os.system("rm {path}".format(
        path=os.path.join(args.output_dir, str(time_id), 'sparse/0/images.txt')
    ))
    os.system("mv {target} {output}".format(
        target=os.path.join(args.output_dir, str(time_id), 'sparse/0/new_images.txt'),
        output=os.path.join(args.output_dir, str(time_id), 'sparse/0/images.txt')
    ))