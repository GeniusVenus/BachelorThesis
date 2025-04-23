# Training

python scripts/train.py --model deeplabv3plus --backbone resnet50  --loss cross_entropy --epochs 50 --checkpoint naver_cross_entropy_resnet50_deeplabv3plus --dataset naver
python scripts/train.py --model deeplabv3plus --backbone resnet50  --loss dice --epochs 50 --checkpoint naver_dice_resnet50_deeplabv3plus --dataset naver
python scripts/train.py --model deeplabv3plus --backbone resnet50  --loss focal --epochs 50 --checkpoint naver_focal_resnet50_deeplabv3plus --dataset naver
python scripts/train.py --model deeplabv3plus --backbone resnet50  --loss jaccard --epochs 50 --checkpoint naver_jaccard_resnet50_deeplabv3plus --dataset naver
python scripts/train.py --model deeplabv3plus --backbone resnet50  --loss lovasz --epochs 50 --checkpoint naver_lovasz_resnet50_deeplabv3plus --dataset naver
python scripts/train.py --model deeplabv3plus --backbone resnet50  --loss tversky --epochs 50 --checkpoint naver_tversky_resnet50_deeplabv3plus --dataset naver











