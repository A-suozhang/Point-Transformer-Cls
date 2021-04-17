dir=$1
gpu=$2

CUDA_VISIBLE_DEVICES=$gpu python train_cls.py --gpu $gpu --learning_rate 0.05 --batch_size 16 --optimizer SGD --log_dir $1 --model mink_transformer --dataset modelnet_voxel --normal 3 --use_voxel --voxel_size 0.05 --epoch 100 
#CUDA_VISIBLE_DEVICES=$gpu python train_cls.py --gpu $gpu --learning_rate 0.05 --batch_size 16 --optimizer SGD --log_dir $1 --model mink_transformer --dataset scannet_voxel --normal 3 --use_voxel --voxel_size 0.05 --epoch 100 

