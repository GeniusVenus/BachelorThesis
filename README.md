# How to run project

## Training script

### Basic training script

```
python scripts/train.py --config experiment_001
```

### Training with overridden parameters

```
python scripts/train.py --config experiment_001 --model deeplabv3plus --batch-size 16 --learning-rate 0.001 --epochs 100
```

## Evaluate script

```
python scripts/inference.py
```

## Inference script

```
python scripts/inference.py --config experiment_001 --model deeplabv3plus --backbone resnet50 --loss all
```
