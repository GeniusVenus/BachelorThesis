# How to run project

## Training script

```
 python scripts/train.py --loss cross_entropy --backbone resnet50 --model deeplabv3plus  --epochs 50 --checkpoint naver_cross_entropy_resnet50_deeplabv3plus --dataset naver
```

## Evaluate script

```
python scripts/evaluation.py --loss cross_entropy --backbone resnet50 --model deeplabv3plus --dataset naver
```

## Inference script

```
python scripts/inference.py --loss cross_entropy --backbone resnet50 --model deeplabv3plus --dataset naver
```
