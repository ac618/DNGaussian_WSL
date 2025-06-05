# Testing Using COLMAP Dataset

## Step 1: Download COLMAP Dataset

```bash
wget https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip
unzip south-building.zip
cd ./south-building/sparse
mkdir 0
mv ./cameras.txt ./0
mv ./images.txt ./0
mv ./points3D.txt ./0
cd ../../
```

## Step 2: Generate monocular depths

```bash
cd dpt
python get_depth_map_for_llff_dtu.py --root_path ../south-building --scenes images
cd ../
mv ./south-building/images/depth_maps ./south-building
```

### TroubleShooting

If encountering the following error:

```bash
Could not load library libcudnn_cnn_infer.so.8. Error: libcuda.so: cannot open shared object file: No such file or directory
Please make sure libcudnn_cnn_infer.so.8 is in your library path!
```

Run the following command:

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

## Step 3: Training

```bash
bash scripts/run_dtu.sh south-building output/dtu/south-building ${gpu_id}
```
