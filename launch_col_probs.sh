CUDA_VISIBLE_DEVICES=3 python train.py --cfg configs/exp1.yml --gpus 0 --output ResNet_1_coll_probs --probs
CUDA_VISIBLE_DEVICES=3 python train.py --cfg configs/exp2.yml --gpus 0 --output ResNet_2_coll_probs --probs
CUDA_VISIBLE_DEVICES=3 python train.py --cfg configs/exp3.yml --gpus 0 --output ResNet_3_coll_probs --probs
CUDA_VISIBLE_DEVICES=3 python train.py --cfg configs/exp4.yml --gpus 0 --output ResNet_4_coll_probs --probs
CUDA_VISIBLE_DEVICES=3 python train.py --cfg configs/exp5.yml --gpus 0 --output ResNet_5_coll_probs --probs
CUDA_VISIBLE_DEVICES=3 python train.py --cfg configs/exp6.yml --gpus 0 --output ResNet_6_coll_probs --probs
CUDA_VISIBLE_DEVICES=3 python train.py --cfg configs/exp7.yml --gpus 0 --output ResNet_7_coll_probs --probs
CUDA_VISIBLE_DEVICES=3 python train.py --cfg configs/exp8.yml --gpus 0 --output ResNet_8_coll_probs --probs