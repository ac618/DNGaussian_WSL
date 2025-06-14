# Testing Using Our Hand Data

## Step 1: Download Data

Link: [EV-ArmData](https://drive.google.com/drive/u/0/folders/1R_9r_xmO2wq8ErYOlbdQ9amSR64hAY1j)

## Step 2: Convert Data into COLMAP Format

```bash
cd ./hand_utils
python data_prepare.py --data_dir $<path to data> --output_dir $<path to processed data>
```

For example:

```bash
cd ./hand_utils
python data_prepare.py --data_dir ../EV-ArmData --output_dir ../hand_test
cd ../
```

## Step 3: Training

```bash
bash scripts/run_dtu.sh $<path to data> $<path to output> ${gpu_id}
```

For example:

```bash
bash scripts/run_dtu.sh hand_test/0 output/dtu/hand/0 0
```

## Step 4: Inference

```bash
bash scripts/inference_dtu.sh $<path to data> $<path to output> ${gpu_id}
```

The rendered results will be stored in folder `$<path to output>/novel`  
For example:

```bash
bash scripts/inference_dtu.sh hand_test/0 output/dtu/hand/0 0
```
