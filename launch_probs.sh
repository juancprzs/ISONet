# exp1
CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/exp1.yml --output exp1_run1_probs --probs
# exp2
CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/exp2.yml --output exp2_run1_probs --probs
CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/exp2.yml --output exp2_run2_probs --probs
CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/exp2.yml --output exp2_run3_probs --probs
# exp3
CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/exp3.yml --output exp3_run1_probs --probs
CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/exp3.yml --output exp3_run2_probs --probs
CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/exp3.yml --output exp3_run3_probs --probs
# exp4
CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/exp4.yml --output exp4_run1_probs --probs
CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/exp4.yml --output exp4_run2_probs --probs
CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/exp4.yml --output exp4_run3_probs --probs
# exp5
CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/exp5.yml --output exp5_run1_probs --probs
CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/exp5.yml --output exp5_run2_probs --probs
CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/exp5.yml --output exp5_run3_probs --probs
# exp6
CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/exp6.yml --output exp6_run1_probs --probs
CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/exp6.yml --output exp6_run2_probs --probs
CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/exp6.yml --output exp6_run3_probs --probs
