# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
from functools import partial
import sys

sys.path.insert(0,os.getcwd())
import torch


from nets.FFNet import resnet

import os
import sys
import numpy as np

import torch.nn as nn
import torch._utils
import torch.nn.functional as F

from nets.FFNet.ffnet_blocks import create_ffnet
from nets.FFNet.model_registry import register_model

model_weights_base_path = None
##########################################################################################
##### 4-Stage GPU FFNets with ResNet backbone.
##### These are trained for use with image sizes of 2048x1024
##### and output a segmentation map of 256x128 pixels
##########################################################################################
@register_model
def segmentation_ffnet150_dAAA():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_A",
        num_classes=19,
        model_name="ffnnet150_dAAA",
        backbone=resnet.Resnet150_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet150/ffnet150_dAAA_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet134_dAAA():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_A",
        num_classes=19,
        model_name="ffnnet134_dAAA",
        backbone=resnet.Resnet134_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet134/ffnet134_dAAA_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet101_dAAA():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_A",
        num_classes=19,
        model_name="ffnnet101_dAAA",
        backbone=resnet.Resnet101_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet101/ffnet101_dAAA_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet86_dAAA():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_A",
        num_classes=19,
        model_name="ffnnet86_dAAA",
        backbone=resnet.Resnet86_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet86/ffnet86_dAAA_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet56_dAAA():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_A",
        num_classes=19,
        model_name="ffnnet56_dAAA",
        backbone=resnet.Resnet56_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet56/ffnet56_dAAA_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet50_dAAA():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_A",
        num_classes=2,
        model_name="ffnnet50_dAAA",
        backbone=resnet.Resnet50_D,
        pre_downsampling=False,
        pretrained_weights_path=None, #都不适用预权重进行寻览 train from scartch
        strict_loading=True,
    )


@register_model
def segmentation_ffnet34_dAAA():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_A",
        num_classes=19,
        model_name="ffnnet34_dAAA",
        backbone=resnet.Resnet34_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet34/ffnet34_dAAA_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet18_dAAA():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_A",
        num_classes=19,
        model_name="ffnnet18_dAAA",
        backbone=resnet.Resnet18_D,
        pre_downsampling=False,
        pretrained_weights_path=None,
        strict_loading=True,
    )


@register_model
def segmentation_ffnet150_dAAC():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet150_dAAC",
        backbone=resnet.Resnet150_D,
        pre_downsampling=False,
        pretrained_weights_path=None,
        strict_loading=True,
    )


@register_model
def segmentation_ffnet86_dAAC():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet86_dAAC",
        backbone=resnet.Resnet86_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet86/ffnet86_dAAC_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet34_dAAC():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet34_dAAC",
        backbone=resnet.Resnet34_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet34/ffnet34_dAAC_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def segmentation_ffnet18_dAAC():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet18_dAAC",
        backbone=resnet.Resnet18_D,
        pre_downsampling=False,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet18/ffnet18_dAAC_cityscapes_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


##########################################################################################
##### Classification models with an FFNet structure. Primarily intended for imagenet
##### initialization of FFNet.
##### See the README for the hyperparameters for training the classification models
##########################################################################################
@register_model
def classification_ffnet150_AAX():
    return create_ffnet(
        ffnet_head_type="A",
        task="classification",
        num_classes=1000,
        model_name="ffnnet150_AAX",
        backbone=resnet.Resnet150,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet150/ffnet150_AAX_imagenet_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def classification_ffnet134_AAX():
    return create_ffnet(
        ffnet_head_type="A",
        task="classification",
        num_classes=1000,
        model_name="ffnnet134_AAX",
        backbone=resnet.Resnet134,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet134/ffnet134_AAX_imagenet_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def classification_ffnet101_AAX():
    return create_ffnet(
        ffnet_head_type="A",
        task="classification",
        num_classes=1000,
        model_name="ffnnet101_AAX",
        backbone=resnet.Resnet101,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet101/ffnet101_AAX_imagenet_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def classification_ffnet86_AAX():
    return create_ffnet(
        ffnet_head_type="A",
        task="classification",
        num_classes=1000,
        model_name="ffnnet86_AAX",
        backbone=resnet.Resnet86,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet86/ffnet86_AAX_imagenet_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def classification_ffnet56_AAX():
    return create_ffnet(
        ffnet_head_type="A",
        task="classification",
        num_classes=1000,
        model_name="ffnnet56_AAX",
        backbone=resnet.Resnet56,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet56/ffnet56_AAX_imagenet_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )



#------------------------------------#
#  TODO  Resnet50 A
#------------------------------------#
@register_model
def classification_ffnet50_AAX():
    return create_ffnet(
        ffnet_head_type="A",
        task="classification",
        num_classes=1000,
        model_name="ffnnet50_AAX",
        backbone=resnet.Resnet50,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet50/ffnet50_AAX_imagenet_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def classification_ffnet34_AAX():
    return create_ffnet(
        ffnet_head_type="A",
        task="classification",
        num_classes=1000,
        model_name="ffnnet34_AAX",
        backbone=resnet.Resnet34,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet34/ffnet34_AAX_imagenet_state_dict_quarts.pth",
        ),
        strict_loading=True,
    )


@register_model
def classification_ffnet18_AAX():
    return create_ffnet(
        ffnet_head_type="A",
        task="classification",
        num_classes=1000,
        model_name="ffnnet18_AAX",
        backbone=resnet.Resnet18,
        pretrained_weights_path=None,
        strict_loading=True,
    )


##########################################################################################
##### This is an example of how these FFNet models would be initialized for training on
##### cityscapes with 2048x1024 images
##########################################################################################
@register_model
def segmentation_ffnet150_dAAC_train():
    return create_ffnet(
        ffnet_head_type="A",
        task="segmentation_C",
        num_classes=19,
        model_name="ffnnet150_dAAC",
        backbone=resnet.Resnet150_D,
        pretrained_weights_path=os.path.join(
            model_weights_base_path,
            "ffnet150/ffnet150_AAX_imagenet_state_dict_quarts.pth",
        ),
        pretrained_backbone_only=True,  # Set when initializing with *FFNet* ImageNet weights to ensure that the head is initialized from scratch
        strict_loading=False,
    )


if __name__ == '__main__':
    
    # 32倍 下采样
    x = torch.randn((1,3,512,512))

    model = segmentation_ffnet50_dAAA()
    out = model(x)

    print(out.shape)


    # def get_fps(net, input_size, device):
    #     net = net.to(device)
    #     h, w = input_size
    #     iterations = None
    #     import time
    #
    #     input = torch.randn(1, 3, h, w).to(device)
    #     with torch.no_grad():
    #         for _ in range(10):
    #             net(input)
    #
    #         if iterations is None:
    #             elapsed_time = 0
    #             iterations = 100
    #             while elapsed_time < 1:
    #                 torch.cuda.synchronize()
    #                 torch.cuda.synchronize()
    #                 t_start = time.time()
    #                 for _ in range(iterations):
    #                     net(input)
    #                 torch.cuda.synchronize()
    #                 torch.cuda.synchronize()
    #                 elapsed_time = time.time() - t_start
    #                 iterations *= 2
    #             FPS = iterations / elapsed_time
    #             iterations = int(FPS * 6)
    #
    #         print('=========Speed Testing=========')
    #         torch.cuda.synchronize()
    #         torch.cuda.synchronize()
    #         t_start = time.time()
    #         for _ in range(iterations):
    #             net(input)
    #         torch.cuda.synchronize()
    #         torch.cuda.synchronize()
    #         elapsed_time = time.time() - t_start
    #         latency = elapsed_time / iterations * 1000
    #     torch.cuda.empty_cache()
    #     FPS = 1000 / latency
    #     print("FPS: ", FPS)
    #
    #     return FPS


    x = torch.randn(1, 3, 512, 512).cuda()
    from nets.USI import flops_parameters

    """
    FLOPs: 16.466G
    Para: 28.544M
    FPS: 144 144.66
    """
    flops_parameters(model.cuda(), input=(x,))

    # outs = net(x)
    # for out in outs:
    #     print(out.size())

    from utils.tools import get_fps

    #
#    from utils.tools import get_fps

    # for i in range(5):
    #
    #  get_fps(net = net,input_size = (512,512),device=torch.device('cuda:0'))
    from datasets.dataloader import BuildDataset, collate_seg
    from torch.utils.data import DataLoader

    path = r'D:\JinKuang\RTSeg\_Aerial_val.txt'
    path1 = r'D:\JinKuang\RTSeg\_M_val.txt'
    data = BuildDataset(train_file=path1, augment=True)
    train_loader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=collate_seg)
    train_iter = iter(train_loader)
    device = torch.device('cuda')
    fps = 0
    for i, batch in enumerate(train_loader):
        image, label, png, (heat2, heat4), \
            ((boundary1, boundary2), boundary4, boundary8, boundary16) = batch
        image = image.to(device)
        fps = max(fps, get_fps(net=model, input=image, device=device))

        print(f'Max_fps: {fps}')

