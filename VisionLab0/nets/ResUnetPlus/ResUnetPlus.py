import torch
import torch.nn as nn
import os,sys

sys.path.insert(0,os.getcwd())
from nets.backbone import resnet50,CBA



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

class Res50UnetPlus(nn.Module):
    def __init__(self,pretrained = False,num_class = 1 + 1,deep_supervision = False):
        super(Res50UnetPlus, self).__init__()
        self.backbone = resnet50(pretrained)
        for i in range(len(self.backbone) - 1):
            setattr(self,f'layer{i + 1}',self.backbone[i])
        self.layer0 = CBA(c = 3,outc = 32,k = 3,p = 1)
        # 3072,1536,768,320
        # 2048 1024 512 256
        in_filter = [ 3072,1536,768,320]
        out_filter = [1024,512,256,64]

        self.deep_supervision = deep_supervision

        if self.deep_supervision:
            self.final1 = nn.Conv2d(64, num_class, kernel_size=1)
            self.final2 = nn.Conv2d(64, num_class, kernel_size=1)
            self.final3 = nn.Conv2d(64, num_class, kernel_size=1)
            self.final4 = nn.Conv2d(64, num_class, kernel_size=1)
        else:
            self.final = nn.Conv2d(64, num_class, kernel_size=1)

        # TODO Unetplus
        self.conv1 = CBA(c =out_filter[-1]//2 + out_filter[-1],outc = out_filter[-1],
                         k = 3,p =  1)

        self.conv1_1 = CBA(c =out_filter[-2] + out_filter[-1],outc = out_filter[-2]//2,
                         k = 3,p =  1)

        self.conv0_2 = CBA(c =out_filter[-2]//2 + out_filter[-1] + out_filter[-1]//2,outc = out_filter[-2]//4,
                         k = 3,p =  1)
        self.conv2_1 = CBA(c =out_filter[1] + out_filter[2],outc = out_filter[2],
                         k = 3,p =  1)

        self.conv1_2 = CBA(c =448,outc = out_filter[-1]*2,
                         k = 3,p =  1)

        self.conv0_3 = CBA(c =288,outc = out_filter[-1],
                         k = 3,p =  1)

        self.conv3_1 = CBA(c =1536,outc = 512,
                         k = 3,p =  1)

        self.conv2_2 = CBA(c =1024,outc = 256,
                         k = 3,p =  1)

        self.conv1_3 = CBA(c =576,outc = 128,
                         k = 3,p =  1)

        self.conv0_4 = CBA(c =352,outc = 64,
                         k = 3,p =  1)


        self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)


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
        x0_0 = self.layer0(x)  #

        x1_0 = self.layer1(x)  # down 2

        x0_1 = self.conv1(torch.cat([ x0_0, self.up(x1_0) ], 1))  # input.size

        x2_0 = self.layer2(x1_0)  # down 2

        x1_1 = self.conv1_1(torch.cat([ x1_0, self.up(x2_0) ], 1))

        x0_2 = self.conv0_2(torch.cat([ x0_0, x0_1, self.up(x1_1) ], 1))

        x3_0 = self.layer3(x2_0)

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))

        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.layer4((x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))

        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


    def freeze_backbone(self):
        for mm in self.backbone:
         for m in mm.parameters():

            m.requires_grad = False

    def unfreeze_backbone(self):
        for mm in self.backbone:
            for m in mm.parameters():
                m.requires_grad = True


if __name__ == '__main__':
    x = torch.randn((1,3,256,256))
    net = Res50UnetPlus(pretrained = False,num_class = 1 + 1,deep_supervision = True)
    out = net(x)
    for o in out:
     print(o.shape)

    from nets.USI import flops_parameters

    """
    FLOPs: 90.822G
    Para: 21.856M
    FPS: 92.52
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


