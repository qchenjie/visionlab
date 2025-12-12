import torch
import torch.nn as nn
# -------------------------------------#
import numpy as np
import torch.optim as optim
from utils.utils_fit import epoch_fit
from utils.logger import Logger
import random
import shutil, os
import platform
from utils.tools import Yam
# ---------------------------------------------------------------#
#                       导入解释器
# ---------------------------------------------------------------#
import argparse

from utils.config import Config
from utils.tools import warm_up, weight__init, fix_seed
from utils.callback import History
from torch.utils.data import DistributedSampler
from datasets.dataloader import MedicalDataset, collate_seg
from torch.utils.data import DataLoader
from utils.summary import summary

# -------------------------------------#
# 更换模型
# -------------------------------------#
from nets.BuildRT import BuildRT
from nets.Bisenet import bisenetv1, bisenetv2
from nets.Pidnet import pidnet
from nets.STDC import model_stage
from nets.lpsnet import get_lspnet_m
from nets.FFNet.ffnet_gpu_small import segmentation_ffnet50_dAAA
from nets.DDRNet.DDRNet_23_slim import get_seg_model
from nets.unet import SegHead

from nets.Unet.Unet import UNet
from nets.UnetPlus.Unetplus import NestedUNet
from nets.ResUnet.ResUnet import Res50Unet
from nets.ResUnetPlus.ResUnetPlus import Res50UnetPlus

from nets.SETR.transformer_seg import SETRModel
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#    TransUnet
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
from nets.TransUnet.vit_seg_modeling import VisionTransformer as ViT_seg
from nets.TransUnet.vit_seg_modeling import CONFIGS
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from nets.Swin_Unet.vision_transformer import SwinUnet
from nets.MIssFormer.MISSFormer import MISSFormer
from nets.TSFormer.TSFormer import TLGFormer as TSFormer
from nets.FHNet.TSFormer import TLGFormer as FHNet

from nets.GCN.GEDNet import GEDNet



#-----------------------------------------------------#
#           TODO?
#-----------------------------------------------------#
from nets.Deeplab.deeplab import DeepLab
from nets.DconnNet import DconnNet
from nets.HiFormer import HiFormer
from nets.HiFormer.configs import get_hiformer_b_configs

from nets.AHF import AHF_Fusion_U_Net
from nets.DAEFormer import daeformer
from nets.PSPnet import PSPnet


# 评估函数
from utils.utils_eval import Eval

if __name__ == '__main__':

    parse_augment = argparse.ArgumentParser(description = 'UDA')

    # add baic augment

    parse_augment.add_argument('--init_epoch', type = int,
                               default = Config[ 'init_epoch' ], help = 'Max epoch...')

    parse_augment.add_argument('--freeze_epoch', type = int,
                               default = Config[ 'freeze_epoch' ], help = 'Max epoch...')

    parse_augment.add_argument('--unfreeze_epoch', type = int,
                               default = Config[ 'unfreeze_epoch' ], help = 'Max epoch...')
    # input_size
    parse_augment.add_argument('--input_shape', '--input', type = tuple,
                               default = Config[ 'input_shape' ], help = 'train image size')

    # batch size
    parse_augment.add_argument('--batch_size', '--bs', type = int,
                               default = Config[ 'bs' ], help = 'train batch size')

    parse_augment.add_argument('--warm_up', type = int, default = 15, help = 'warm_up paramters...')

    # 训练集
    parse_augment.add_argument('--train_txt', '--st', type = str, default = Config[ 'train_txt' ],
                               help = 'train file')

    # 测试集
    parse_augment.add_argument('--test_txt', '-tt', type = str, default = Config[ 'test_txt' ],
                               help = 'test file')

    # 验证集
    parse_augment.add_argument('--val_txt', '--sv', type = str, default = Config[ 'val_txt' ],
                               help = 'val file')
    # 方法
    parse_augment.add_argument('--method', type = str, default = 'ours',
                               required = True, help = 'Compare Method...')
    # 训练模式
    parse_augment.add_argument('--mode', '--md', type = str, default = 'co-train',
                               help = 'select train mode')

    # DDP train
    parse_augment.add_argument('--distributed', '--dist', action = 'store_true',
                               default = False, help = 'DDP training...')

    # num_worker
    parse_augment.add_argument('--num_worker', '--nw', type = int, default = Config[ 'num_worker' ],
                               help = 'num_worker...')

    # decay rate
    parse_augment.add_argument('--decay_rate', '--d', type = float, default = Config[ 'decay_rate' ])

    # momentum
    parse_augment.add_argument('--momentum', '--m', type = float, default = Config[ 'momentum' ])

    # optimizer type
    parse_augment.add_argument('--optimizer_type', '--op', type = str, default = Config[ 'optimizer' ])

    # min lr
    parse_augment.add_argument('--min_lr', '--mlr', type = float, default = Config[ 'min_lr' ])

    # model_path
    parse_augment.add_argument('--model_path', '--model', type = str, default = Config[ 'model_path' ])

    # adam max lr
    parse_augment.add_argument('--adam_max_lr', '--adam_lr', type = float, default = Config[ 'adam_max_lr' ])

    # sgd max lr
    parse_augment.add_argument('--sgd_max_lr', '--sgd_lr', type = float, default = Config[ 'sgd_max_lr' ])

    # seed
    parse_augment.add_argument('--seed', '--s', type = int, default = 1234)

    # save step
    parse_augment.add_argument('--save_step', '--save', type = int, default = Config[ 'save_step' ],
                               help = 'save model step')
    # dataset name
    parse_augment.add_argument('--datasetname', type = str, required = True, help = 'datasetname...')
    # accumulate
    parse_augment.add_argument('--accumulate', '--ac', type = int,
                               default = 4, help = 'accumulate grad...')
    # class
    parse_augment.add_argument('--classes', '--c', type = tuple, default = Config[ '_Class' ])
    # 解释
    args = parse_augment.parse_args()

    # -------------------------------------------------------------------------------#
    #                              多卡训练
    # -------------------------------------------------------------------------------#
    platformer = platform.system()

    if args.distributed:
        backendsname = 'gloo' if platformer == 'Windows' else 'nccl'

        # 初始化后端
        torch.distributed.init_process_group(backend = backendsname)
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0  # 设置

    # ------------------------------------------------------------------------------#
    #                               参数赋值
    # ------------------------------------------------------------------------------#
    init_epoch = args.init_epoch
    freeze_epoch = args.freeze_epoch
    unfreeze_epoch = args.unfreeze_epoch

    num_worker = args.num_worker

    freeze_batch = args.batch_size

    input_shape = args.input_shape

    # ---------------------------------------------------#
    #                 训练标签
    # ---------------------------------------------------#
    train_txt = args.train_txt
    test_txt = args.test_txt
    val_txt = args.val_txt

    class_num = len(args.classes) + 1

    model_path = args.model_path

    save_step = args.save_step

    min_lr = args.min_lr

    warm_iter = args.warm_up

    # ------------------------------#
    # 优化器种类
    # ------------------------------#
    optimizer_type = args.optimizer_type

    max_lr = args.adam_max_lr if optimizer_type == 'adam' else args.sgd_max_lr

    decay_rate = args.decay_rate
    momentum = args.momentum

    # fix seed
    seed = args.seed

    # datasetname
    datasetname = args.datasetname

    fix_seed(seed)

    # Evaluate
    eval = Eval(class_num = class_num)

    # ---------------------------------#
    #            权重保存路径
    # ---------------------------------#
    # 对比方法
    method = args.method
    save_path = f'Pth/{datasetname}/{method}'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 设置多卡
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')

    # -------------------------------------------#
    #                初始化模型
    # -------------------------------------------#

    if (method == 'SFGNet'):
        model = BuildRT(n_classes = class_num)

    elif method == 'bisenetv1':
        model = bisenetv1.BiSeNetV1(class_num)

    elif method == 'bisenetv2':
        model = bisenetv2.BiSeNetV2(class_num)

    elif method == 'pidnet':
        model = pidnet.get_pred_model(name = 'pidnet_s', num_classes = class_num)

    elif method == 'ddrnet':
        model = get_seg_model(pretrained = False, num_classes = class_num, augment = False)

    elif method == 'ffnet':
        model = segmentation_ffnet50_dAAA()

    elif method == 'stdc':
        model = model_stage.BiSeNet('STDCNet813', class_num, use_boundary_8 = True)

    elif method == 'lpsnet':
        model = get_lspnet_m()

    elif method == 'unet':
        model = UNet(n_channels = 3, n_classes = class_num, bilinear = True)

    elif method == 'unetplus':
        # double
        model = NestedUNet(num_classes = class_num, deep_supervision = True)

    elif method == 'resunet':
        model = Res50Unet(pretrained = True, num_class = class_num)

    elif method == 'resunetplus':
        model = Res50UnetPlus(pretrained = True, num_class = class_num, deep_supervision = True)

    elif method == 'transunet':
        config_vit = CONFIGS[ 'R50-ViT-B_16' ]
        config_vit.n_classes = class_num
        config_vit.n_skip = 3

        img_size = input_shape[ 0 ]
        vit_patches_size = 16
        # if config_vit.vit_name.find('R50') != -1:
        #     config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        model = ViT_seg(config_vit, img_size = img_size, num_classes = config_vit.n_classes).cuda()
        model.load_from(weights = np.load(r'./nets/TransUnet/checkpoints/imagenet21k+imagenet2012_R50+ViT-B_16.npz'))

    elif method == 'setr':
        # 没有预训练权重
        model = SETRModel(patch_size = (32, 32),
                          in_channels = 3,
                          out_channels = class_num,
                          hidden_size = 1024,
                          sample_rate = 5,
                          num_hidden_layers = 4,
                          num_attention_heads = 16,
                          decode_features = [ 512, 256, 128, 64 ])

    elif method == 'swin_unet':

        img_size = input_shape[ 0 ]
        num_classes = class_num
        # 配置文件
        config = {'img_size': img_size,
                  'embed_dim': 96,
                  'depths': [ 2, 2, 2, 2 ],
                  'num_heads': [ 3, 6, 12, 24 ],
                  'window_size': 7,
                  'patch_size': 4,
                  'in_chans': 3,  # 输入通道数
                  'mlp_ratio': 4.,
                  'qkv_bias': True,
                  'qk_scale': None,
                  'ape': False,
                  'drop_rate': 0.0,
                  'patch_norm': True,
                  'drop_path_rate': 0.2,
                  'use_checkpoint': False,
                  # 预训练权重
                  'pretrained_path': r"./nets/Swin_Unet/checkpoints/swin_tiny_patch4_window7_224.pth"
                  }

        yaml = Yam(config)

        # print(yaml.img_size)

        model = SwinUnet(config = yaml, img_size = img_size, num_classes = num_classes)
        model.load_from()  # 加载预训练模型

    elif method == 'missformer':  # MISSFormer
        model = MISSFormer(num_classes = class_num, img_size = input_shape[ 0 ])

    elif method == 'NAFormer':
        model = TSFormer(num_classes = class_num, img_size = input_shape[ 0 ], pretrained = True)


    elif method == 'gcn':
        model = GEDNet(n_channels = 3, num_classes = class_num)
  

    elif method == 'AFDSeg':

        model = FHNet(num_classes = class_num, img_size = input_shape[ 0 ])

    elif method == 'deeplabv3':
        model = DeepLab(pretrain = True,
                        num_classes = class_num)

    elif method == 'pspnet':
        model = PSPnet.PSPNet(num_classes = class_num)

    elif method == 'dconnNet':
        model = DconnNet.DconnNet(num_class = class_num)

    elif method == 'Hiformer':
        config = get_hiformer_b_configs()
        model = HiFormer.HiFormer(config = config,
                                  img_size = input_shape[0],
                                  n_classes = class_num)

    elif method == 'ahfunet':
        model = AHF_Fusion_U_Net.UNetAFF(n_channels = 3,
                                         n_classes = class_num)
    elif method == 'daeformer':
        model = daeformer.DAEFormer(num_classes = class_num)
    else:
        raise NotImplementedError('No exist Method!!!')

    # -------------------------------------------#
    # 初始化日志器
    # -------------------------------------------#
    logger = Logger(file_path = os.path.join(save_path, 'looger'))

    model_path = args.model_path

    if os.path.exists(model_path) & (model_path != ''):

        logger.info(f'Load>>>>>>>>>>Pretrained Model: {model_path}')

        miss = [ ]  # 缺少模块
        exist = [ ]  # 存在模块
        net = model.state_dict()  # 当前模型的图结构

        pretrainnet = torch.load(model_path, device)  # 预训练结构

        temp = {}  # 收集共有层
        for k, v in pretrainnet.items():

            if (k in net.keys()) and  (np.shape(net[ k ]) == np.shape(v)):
                temp[ k ] = v
                exist.append(k)
            else:
                miss.append(k)
        net.update(temp)
        # 导入
        model.load_state_dict(net)

    # if not os.path.exists(model_path):
    #     weight__init(model)  # 初始化网络参数
    #     logger.info('没有导入预训练权重，初始化网络参数')

    # --------------------------------------------#
    # 优化器
    # --------------------------------------------#

    modeloptimizer = {
        'adam': optim.AdamW(model.parameters(), lr = max_lr, weight_decay = decay_rate),
        'sgd': optim.SGD(model.parameters(), lr = max_lr, momentum = momentum, weight_decay = decay_rate)
    }[ optimizer_type ]

    # modeloptimizer = optim.SGD(model.optim_parameters(2.5e-4), lr = 2.5e-4, momentum = 0.9,
    #                              weight_decay = 0.0005)

    # --------------------------------------------#
    # 数据加载
    # --------------------------------------------#
    trainseg = MedicalDataset(input_shape = input_shape, train_file = train_txt, num_classes = class_num, augment = True)

    trainSegData = DataLoader(trainseg, batch_size = freeze_batch, num_workers = num_worker, collate_fn = collate_seg,
                              shuffle = True, pin_memory = True)

    # 测试集
    testseg = MedicalDataset(input_shape = input_shape, train_file = test_txt,
                           num_classes = class_num)
    testSegData = DataLoader(testseg, batch_size = freeze_batch, num_workers = num_worker, collate_fn = collate_seg,
                             shuffle = False)

    # 验证集
    # valseg = BuildDataset(input_shape = input_shape, train_file = val_txt,
    #                        num_classes = class_num)
    # valSegData = DataLoader(valseg, batch_size = freeze_batch, num_workers = num_worker, collate_fn = collate_seg,
    #                          shuffle = False)

    # wait for all porcesses to synchronize
    if args.distributed:
        torch.distributed.barrier()

    # 训练
    for epoch in range(init_epoch, unfreeze_epoch):

        if args.distributed:
            trainSegData.batch_sampler.sampler.set_epoch(epoch)

        # -----------------------------------#
        # 使用余弦退火法
        # -----------------------------------#
        warm_up(modeloptimizer, freeze_epoch, epoch,
                min_lr, max_lr, warm_iter)

        epoch_fit(epoch, unfreeze_epoch, save_step,
                  model, modeloptimizer, [ trainSegData, testSegData, None ],
                  device, logger, args, save_path, eval, local_rank)




