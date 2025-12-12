import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.Transim.sample import enhance_net_nopool,kl_loss,mse_loss


import sys,os


sys.path.insert(0,os.getcwd())


from nets.backbone import CBA


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
                            CBA(c = in_channels,outc = out_channels,k = 3,p=1),
                            CBA(c = out_channels, outc = out_channels, k = 3, p = 1),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        #input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()

        self.net1 = enhance_net_nopool()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1536, 512, bilinear)
        self.up2 = Up(768, 256, bilinear)
        self.up3 = Up(384, 128, bilinear)
        self.up4 = Up(192, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.boundary = OutConv(64,1)
        self.points = OutConv(128,1)

    def forward(self, x):

        # restruction
        recon = self.net1(x)


        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)


        points = self.points(x)
        x = self.up4(x, x1)
        logits = self.outc(x)
        boundary = self.boundary(x)

        return logits,boundary,points,recon


    def _forward(self,x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)

        logits = self.outc(x)

        return logits

if __name__ == '__main__':
    x = torch.randn((1,3,256,256))
    net = UNet(n_channels=3, n_classes=2,bilinear = True)
    out = net(x)

    for o in out:
     print(o.shape)

    # from nets.USI import flops_parameters
    #
    # """
    # FLOPs:55.989G
    # Para: 31.385M
    # FPS: 161.73
    # """
    # flops_parameters(net, input=(x,))
    #
    # # net.get_params()
    #
    # from utils.tools import get_fps
    #
    # # for i in range(5):
    # #
    # #  get_fps(net = net,input_size = (512,512),device=torch.device('cuda:0'))
    # from datasets.dataloader import BuildDataset, collate_seg
    # from torch.utils.data import DataLoader
    #
    # path = r'D:\JinKuang\RTSeg\_Aerial_val.txt'
    # path1 = r'D:\JinKuang\RTSeg\_M_val.txt'
    # data = BuildDataset(train_file=path1, augment=True)
    # train_loader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=collate_seg)
    # train_iter = iter(train_loader)
    # device = torch.device('cuda')
    # fps = 0
    # for i, batch in enumerate(train_loader):
    #     image, label, png, (heat2, heat4), \
    #         ((boundary1, boundary2), boundary4, boundary8, boundary16) = batch
    #     image = image.to(device)
    #
    #     image = F.interpolate(image,size=(256,256),mode='bilinear',align_corners=True)
    #     fps = max(fps, get_fps(net=net, input=image, device=device))
    #
    #     print(f'Max_fps: {fps}')
