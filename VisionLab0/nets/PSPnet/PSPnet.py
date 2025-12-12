import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from nets.backbone import resnet50
warnings.filterwarnings('ignore')

class PSPBlock(nn.Module):
    def __init__(self,inc):
        super(PSPBlock, self).__init__()
        self.pool1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels = inc,out_channels = inc//4,kernel_size = 1,padding = 1),
            nn.BatchNorm2d(inc//4),
            nn.ReLU(inplace = True)
        )
        self.pool2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            nn.Conv2d(in_channels = inc, out_channels = inc // 4, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(inc // 4),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = inc // 4, out_channels = inc // 4, kernel_size = 1)
        )
        self.pool3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(3),
            nn.Conv2d(in_channels = inc, out_channels = inc // 4, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(inc // 4),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = inc // 4, out_channels = inc // 4, kernel_size = 1)
        )
        self.pool4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(6),
            nn.Conv2d(in_channels = inc, out_channels = inc // 4, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(inc // 4),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = inc // 4, out_channels = inc // 4, kernel_size = 1)
        )

        self.outc = nn.Sequential(nn.Conv2d(in_channels = inc//4 * 5 + inc,
                                            out_channels = inc,kernel_size = 3,
                                            padding = 1),
                                  nn.BatchNorm2d(inc),
                                  nn.ReLU(inplace = True))
    def forward(self,x):
        bs,ct,h,w = x.shape

        layer1 = self.pool1(x)
        layer1 = F.upsample(layer1,(h,w),align_corners = True,mode = 'bilinear')
        layer2 = self.pool2(x)
        layer2= F.upsample(layer2, (h, w),align_corners = True,mode = 'bilinear')
        layer3 = self.pool3(x)
        layer3 = F.upsample(layer3, (h, w),align_corners = True,mode = 'bilinear')
        layer4 = self.pool4(x)
        layer4 = F.upsample(layer4, (h, w),align_corners = True,mode = 'bilinear')

        x = torch.cat((x,layer1,layer2,layer3,layer3,layer4),dim = 1)


        return self.outc(x)


class PSPNet(nn.Module):
    def __init__(self,pretrained = True,num_classes = 2):
        super(PSPNet, self).__init__()

        self.backbone = nn.ModuleList([ ])

        for layer in resnet50(pretrained = pretrained):
            self.backbone.append(layer)

        self.fin = PSPBlock(2048)

        self.out = nn.Sequential(nn.Conv2d(in_channels = 2048,out_channels = 256,
                             kernel_size = 3,padding = 1),
                                 nn.BatchNorm2d(256),
                                 nn.ReLU(inplace = True),
                                 nn.Conv2d(in_channels = 256,out_channels = num_classes,
                                           kernel_size = 3,padding = 1))


    def forward(self,x):

        b,c,h,w = x.shape

        for layer in self.backbone:
         x = layer(x)

        x = self.fin(x)
        x = self.out(x)
        x = F.interpolate(x,(h,w),align_corners = True,mode = 'bilinear')

        return x

if __name__ == '__main__':
    x = torch.randn((1,3,256,256))
    net = PSPNet(num_classes = 2)
    out = net(x)
    print(out.shape)