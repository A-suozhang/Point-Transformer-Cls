gpu=$1

CUDA_VISIBLE_DEVICES=$gpu python train_cls.py --gpu $gpu --learning_rate 0.05 --batch_size 64 --optimizer SGD --log_dir with_BN_LN --normal
# CUDA_VISIBLE_DEVICES=$gpu python train_cls.py --gpu $gpu --learning_rate 0.001 --batch_size 64 --optimizer Adam --log_dir adam --normal
# CUDA_VISIBLE_DEVICES=$gpu python train_cls.py --gpu $gpu --learning_rate 0.05 --batch_size 4 --optimizer SGD --log_dir debug --normal
