#!/bin/bash

models=(
    "alexnet"
    "convnext_tiny"
    "convnext_base"
    "densenet121"
    "densenet169"
    "efficientnet_b0"
    "efficientnet_b4"
    "efficientnet_v2_m"
    "googlenet"
    "inception_v3"
    "maxvit_t"
    "mobilenet_v2"
    "mobilenet_v3_small"
    "regnet_y_400mf"
    "resnet18"
    "resnet50"
    "resnext50_32x4d"
    "shufflenet_v2_x1_0"
    "squeezenet1_0"
    "swin_s"
    "swin_v2_s"
    "vgg16"
    "vgg16_bn"
    "vit_b_16"
    "vit_l_16"
    "wide_resnet50_2"
)


for model in "${models[@]}"; do
    python train.py --model=$model
done