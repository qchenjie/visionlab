import torch
import torch.nn as nn
from nets.backbone import resnet50
class DeepLab(nn.Module):
    def __init__(self,pretrain = False,num_classes = 2):
        super(DeepLab, self).__init__()


        self.backbone = nn.ModuleList([])

        for layer in resnet50(pretrained = pretrain):
            self.backbone.append(layer)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 512,
                      kernel_size = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True)
        )
        self.aspp = ASPP(inc = 1024,outc = 1024)
        self.decoder = Decoder(num_classes)
    def forward(self,x):

        for stage,layer in enumerate(self.backbone[:-1]):

            if stage == 1:
                short_cut = self.conv(layer(x))
            x = layer(x)

        x = self.aspp(x)

        x = self.decoder(short_cut,x)
        return x


class ASPP(nn.Module):
    def __init__(self,inc,outc):
        super(ASPP, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = inc,out_channels = outc,
                      kernel_size = 1),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace = True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = inc, out_channels = outc,
                      kernel_size = 3,padding = 6,dilation = 6),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace = True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = inc, out_channels = outc,
                      kernel_size = 3, padding = 12, dilation = 12),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace = True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = inc, out_channels = outc,
                      kernel_size = 3, padding = 18, dilation = 18),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace = True)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.out = nn.Conv2d(in_channels = 5*outc,out_channels = outc,kernel_size = 1)


    def forward(self,x):

        layer1 = self.conv1(x)
        layer2 = self.conv2(x)
        layer3 = self.conv3(x)
        layer4 = self.conv4(x)
        b,c,h,w = layer4.shape
        layer5 = self.pool(x)
        layer5 = layer5.repeat((1,1,h,w))

        x = torch.cat((layer1,layer2,layer3,layer4,layer5),
                      dim = 1)
        x = self.out(x)

        return x

class Decoder(nn.Module):
    def __init__(self,num_classes = 2):
        super(Decoder, self).__init__()
        self.up = nn.Upsample(scale_factor = 4)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = 1024, out_channels = 512,
                      kernel_size = 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 1024, out_channels = 512,
                      kernel_size = 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels = 512, out_channels = 256,
                      kernel_size = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True)
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor = 4),
            nn.Conv2d(in_channels = 256, out_channels = 128,
                      kernel_size =1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True)
        )
        self.out = nn.Conv2d(in_channels = 128,out_channels = num_classes ,
                             kernel_size = 1)
    def forward(self,x1,x2):
        x2 = self.up(x2)
        x2 = self.conv(x2)

        x = torch.cat((x1,x2),dim = 1)
        x = self.conv1(x)
        x = self.up1(x)

        x = self.out(x)
        return x

if __name__ == '__main__':
    from torchsummary.torchsummary import summary

    x = torch.randn((1,3,224,224))
    net = DeepLab()

    for k,v in net.named_parameters():
        print(k,v.shape)
    # out = net(x)
    # #summary(net,(3,512,512))
    # print(out.shape)



