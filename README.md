### Our code is based on pytorch-image-models and our models are based on DeiT.

## Data Preparation
Download and extract ImageNet train and validation datasets.
```
/path/to/imagenet/
  train/
    class_1/
      img1.jpeg
      ...
    class_2/
      img2.jpeg
      ...
  val/
    class_1/
      img3.jpeg
    ...
    class_2/
      img4.jpeg
    ...
```

## Training
Set the path and  change the "GPU nums", "--model", "--reduce_ratio" parameters.
```
./distributed_train.sh <GPU nums=4> /path/to/imageNet/ --model deit_small_patch16_224 --reduce_ratio <70> --GCM --gp avg --L1st --GCAM --dftGF --drop-path 0.1 --model-ema-decay 0.99996 --opt adamw --opt-eps 1e-8 --weight-decay 0.05 --lr 1e-3 --warmup-lr 1e-6 --min-lr 1e-5 --decay-epochs 30 --warmup-epochs 5 --aa rand-m9-mstd0.5-inc1 --train-interpolation bicubic --reprob 0.25 --mixup 0.8 --cutmix 1.0  --epochs 300 --sched cosine -b 512 -j 24 --log-wandb --amp --output /home/ubuntu/timm/output/train/dctGF --experiment small 
```

## Evaluation
Set the path and change the "checkpoint" and "reduce_ratio" parameters
(Because of rule that submissions should not contain pointers to supplementary material on the web and limit of file capacity, weight could not be attached. Even if we compress the weight file, it is over 100MB.)
```
python validate.py /path/to/imageNet/ --model deit_tiny_patch16_224 --checkpoint /path/to/weights/small/<weights file> --reduce_ratio <70> --dctGF_1d --GCM --gp avg --L1st --GCAM
```

### We provide DCT-ViT-Small/Tiny models results and pretrained weights on ImageNet-1k.

| Model | Keep rate | Acc@1 | Acc@5 | MACs (G) | #Params |
| --- | --- | --- | --- | --- | --- |
| DeiT-Small | - | 79.8 | 95.0 | 4.6 | 22.0M |
| DCT-ViT-Small | 1.0 | 80.24 | 94.94 | 4.04 | 21.5M |
| DCT-ViT-Small | 0.9 | 79.96 | 94.69 | 3.47 | 21.3M |
| DCT-ViT-Small | 0.8 | 79.55 | 94.61 | 2.97 | 21.0M |
| DCT-ViT-Small | 0.7 | 79.08 | 94.30 | 2.55 | 20.8M |
| DCT-ViT-Small | 0.6 | 78.51 | 94.04 | 2.2 | 20.7M |
| DCT-ViT-Small | 0.5 | 77.92 | 93.50 | 1.9 | 20.5M |

| Model | Keep rate | Acc@1 | Acc@5 | MACs (G) | #Params |
| --- | --- | --- | --- | --- | --- |
| DeiT-Tiny | - | 72.2 | 91.1 | 1.3 | 5.0M |
| DCT-ViT-Tiny | 1.0 | 73.27 | 91.59 | 1.096 | 6.0M |
| DCT-ViT-Tiny | 0.9 | 72.91 | 91.49 | 0.938 | 5.9M |
| DCT-ViT-Tiny | 0.8 | 72.71 | 91.22 | 0.8 | 5.8M |
| DCT-ViT-Tiny | 0.7 | 72.37 | 90.91 | 0.688 | 5.7M |
| DCT-ViT-Tiny | 0.6 | 71.86 | 90.61 | 0.593 | 5.6M |
| DCT-ViT-Tiny | 0.5 | 71.53 | 90.43 | 0.514 | 5.5M |

## Weigts
[1D_DCT_Pruning_Ratio_90](https://seoultechackr-my.sharepoint.com/:u:/g/personal/jhlees_seoultech_ac_kr/Ea5Q0uzh_g9OujRQSMQbbqsBaqKxc0cH2GLLakZYIHrEPA?e=6CUimY)

[1D_DCT_Pruning_Ratio_80](https://seoultechackr-my.sharepoint.com/:u:/g/personal/jhlees_seoultech_ac_kr/EdwO-MTFPm5Og6LydYtspPsBTDoAJSvaRIi4i3nnbRMQYA?e=82WEfA)

[1D_DCT_Pruning_Ratio_70](https://seoultechackr-my.sharepoint.com/:u:/g/personal/jhlees_seoultech_ac_kr/ERhJwHNSXvNLkrV-ftU56iwBB2iGilqUNZCAGJuG2YHpmA?e=4SDrj1)

[Tiny_Ratio_100_best](https://seoultechackr-my.sharepoint.com/:u:/g/personal/jhlees_seoultech_ac_kr/EaZUVzC_Uk9IitJ3nLOHtSMBOZ_LQevZCPPphUrwPWsEiw?e=xfa8LI)

[Small_Ratio_100_best](https://seoultechackr-my.sharepoint.com/:u:/g/personal/jhlees_seoultech_ac_kr/EUEBNolUjwVAnuWKCUQ60RcBziKyJW3nyc2klto5OXA_tA?e=fTbrFE)
