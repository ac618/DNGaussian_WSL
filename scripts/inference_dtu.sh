dataset=$1 
workspace=$2 
export CUDA_VISIBLE_DEVICES=$3

python inference.py -s $dataset --model_path $workspace --dataset DTU --eval --rand_pcd \
                    --n_sparse 4 --skip_train