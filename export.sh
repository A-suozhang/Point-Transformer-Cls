dir=$1
gpu=$2

#CUDA_VISIBLE_DEVICES=$gpu python train_cls.py --gpu $gpu --learning_rate 0.05 --batch_size 16 --optimizer SGD --log_dir $1 --model pct_seg --dataset scannet --normal 6 --num_point 4096 --epoch 100 --pretrain --mode export 
CUDA_VISIBLE_DEVICES=$gpu python train_cls.py --gpu $gpu --learning_rate 0.05 --batch_size 16 --optimizer SGD --log_dir $1 --model pct_seg_instance --dataset scannet --normal 6 --num_point 4096 --epoch 100 --pretrain --mode export --aux seg

