import torch
import numpy as np

import os,sys


sys.path.insert(0,os.getcwd())


from nets.TransUnet.vit_seg_modeling import VisionTransformer as ViT_seg

from nets.TransUnet.vit_seg_modeling import CONFIGS

if __name__ == "__main__":
    config_vit = CONFIGS[ 'R50-ViT-B_16' ]
    config_vit.n_classes = 2
    config_vit.n_skip = 3

    img_size = 256
    vit_patches_size = 16
    # if config_vit.vit_name.find('R50') != -1:
    #     config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    net = ViT_seg(config_vit, img_size = img_size, num_classes = config_vit.n_classes).cuda()
    #net.load_from(weights = np.load(r'./TransUnet/checkpoints/imagenet21k+imagenet2012_R50+ViT-B_16.npz'))
    x = torch.randn((1, 3, 256, 256)).cuda()

    out = net(x)

    print(out.shape)

    from nets.USI import flops_parameters

    """
    FLOPs: 32.238
    Para: 93.231M
    FPS: 51.75
    """
    flops_parameters(net, input=(x,))

    # net.get_params()

    from utils.tools import get_fps

    # for i in range(5):
    #
    #  get_fps(net = net,input_size = (512,512),device=torch.device('cuda:0'))
    from datasets.dataloader import BuildDataset, collate_seg
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

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

        image = F.interpolate(image, size=(256, 256), mode='bilinear', align_corners=True)
        fps = max(fps, get_fps(net=net, input=image, device=device))

        print(f'Max_fps: {fps}')

