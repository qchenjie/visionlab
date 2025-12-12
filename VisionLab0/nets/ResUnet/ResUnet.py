import torch
import torch.nn as nn
import os,sys

sys.path.insert(0,os.getcwd())
from nets.backbone import resnet50



class UP(nn.Module):
    def __init__(self,inc,outc):
        super(UP, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.conv1 = nn.Conv2d(in_channels = inc,out_channels = outc,kernel_size = 3,padding = 1)
        self.relu1 = nn.ReLU(inplace = True)

        self.out = nn.Sequential(nn.Conv2d(in_channels = outc, out_channels = outc, kernel_size = 3, padding = 1),
                                 nn.ReLU(inplace = True))

    def forward(self,x,x1):
        #--------------------------#
        # x1降采样
        #--------------------------#
        outputs = torch.cat((x1,self.up(x)),dim = 1)
        outputs = self.conv1(outputs)
        outputs = self.relu1(outputs)
        return self.out(outputs)

class Res50Unet(nn.Module):
    def __init__(self,pretrained = False,num_class = 1 + 1):
        super(Res50Unet, self).__init__()
        self.backbone = resnet50(pretrained)
        for i in range(len(self.backbone)):
            setattr(self,f'layer{i + 1}',self.backbone[i])
        # 3072,1536,768,320
        # 2048 1024 512 256
        in_filter = [ 3072,1536,768,320]
        out_filter = [1024,512,256,64]

        for i in range(len(in_filter)):
            setattr(self,f'up{i + 1}',UP(in_filter[i],out_filter[i]))


        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor = 2),
            nn.Conv2d(in_channels = 64,out_channels = 64,kernel_size = 3,padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64,out_channels = 64,kernel_size = 3,padding = 1),
            nn.ReLU()
        )

        self.out = nn.Conv2d(64,num_class,kernel_size = 3,padding = 1,bias = False)

    def forward(self,x):
        #--------------------------------#
        # 降采样
        #--------------------------------#

        feat1 = self.layer1(x) #第一层
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        feat5 = self.layer5(feat4)


        #--------------------------------#
        # 上采样
        #--------------------------------#

        up1 = self.up1(feat5,feat4)
        up2 = self.up2(up1,feat3)
        up3 = self.up3(up2,feat2)
        up4 = self.up4(up3,feat1)



        #--------------------------------#
        # output
        #--------------------------------#
        out = self.up_conv(up4)

        out = self.out(out)
        return out

    def freeze_backbone(self):
        for mm in self.backbone:
         for m in mm.parameters():

            m.requires_grad = False

    def unfreeze_backbone(self):
        for mm in self.backbone:
            for m in mm.parameters():
                m.requires_grad = True


if __name__ == '__main__':
    x = torch.randn((1,3,512,512))
    net = Res50Unet(pretrained = False,num_class = 1 + 1)
    out = net(x)
    print(out.shape)

    from nets.USI import flops_parameters

    """
    FLOPs: 172.205G
    Para: 73.353M
    FPS: 124.91
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


