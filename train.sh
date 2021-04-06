dir=$1
gpu=$2

# CUDA_VISIBLE_DEVICES=$gpu python train_cls.py --gpu $gpu --learning_rate 0.2 --batch_size 64 --optimizer SGD --log_dir $1 --normal 3
# CUDA_VISIBLE_DEVICES=$gpu python train_cls.py --gpu $gpu --learning_rate 0.001 --batch_size 32 --optimizer Adam --log_dir $1 --dataset scanobjnn --decay_rate 3.e-4
# CUDA_VISIBLE_DEVICES=$gpu python train_cls.py --gpu $gpu --learning_rate 0.05 --batch_size 4 --optimizer SGD --log_dir debug --normal --dataset modelnet_voxel --model mink --use_voxel --voxel_size 0.02
# CUDA_VISIBLE_DEVICES=$gpu python train_cls.py --gpu $gpu --learning_rate 0.05 --batch_size 16 --optimizer SGD --log_dir debug --normal 3
# CUDA_VISIBLE_DEVICES=$gpu python train_cls.py --gpu $gpu --learning_rate 0.05 --batch_size 8 --optimizer SGD --log_dir $1 --model pct_seg --dataset scannet --normal 6 --num_point 8192 --pretrain --epoch 100
CUDA_VISIBLE_DEVICES=$gpu python train_cls.py --gpu $gpu --learning_rate 0.05 --batch_size 16 --optimizer SGD --log_dir $1 --model pct_seg --dataset scannet --normal 6 --num_point 4096 --pretrain --epoch 300 --eval_only
# CUDA_VISIBLE_DEVICES=$gpu python train_cls.py --gpu $gpu --learning_rate 0.05 --batch_size 16 --optimizer SGD --log_dir debug --dataset scanobjnn

