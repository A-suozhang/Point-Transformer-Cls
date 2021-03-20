gpu=$1

# CUDA_VISIBLE_DEVICES=$gpu python train_cls.py --gpu $gpu --learning_rate 0.1 --batch_size 32 --optimizer SGD --log_dir test_0
CUDA_VISIBLE_DEVICES=$gpu python train_cls.py --gpu $gpu --learning_rate 0.05 --batch_size 64 --optimizer SGD --log_dir test_3
