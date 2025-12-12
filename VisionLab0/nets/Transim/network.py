import sys,os


sys.path.insert(0,os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.Transim.sample import CSDN_Tem,\
                                CSDN_Temd,Hist_adjust


class CBA(nn.Module):
    def __init__(self, in_planes, out_planes, kernel = 3, stride = 1):
        super(CBA, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size = kernel, stride = stride,
                      padding = kernel // 2, bias = False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):

        return (self.layer(x))

class SE(nn.Module):
    def __init__(self,channels,ratio=16):
        super(SE, self).__init__()

        self.maxpool = nn.AdaptiveMaxPool2d((1,1))
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))


        self.conv = nn.Conv2d(in_channels = channels * 2,
                              out_channels = channels,
                              kernel_size = 1)

        #self.alpha = nn.Parameter(torch.tensor(0.995),requires_grad = True)
        self.fc=nn.Sequential(
            nn.Linear(channels,channels//ratio,False),
            nn.ReLU(),
            nn.Linear(channels//ratio, channels, False),
            nn.Sigmoid()
        )


    def forward(self,x):


        b,c,_,_ = x.size()

        avg = self.avgpool(x)#.view(b,c)
        max = self.maxpool(x)#.view(b,c)

        out = self.conv(torch.cat((avg,max),dim = 1))
       


        out2 = self.fc(out.view(b,c)).view(b,c,1,1)

       

        return x * out2.expand_as(x)  #TODO?





class LowFR(nn.Module):
    def __init__(self,inplanes = None,outplanes = None,ksize = 1,stride = 1):
        super(LowFR, self).__init__()

        if outplanes is None: outplanes = inplanes
        self.conv1 = CBA(in_planes = inplanes,out_planes = outplanes,
                         kernel = ksize,stride = stride)

        self.relu = nn.ReLU(inplace = True)

        self.bn = nn.BatchNorm2d(outplanes)

        self.conv2 = nn.Conv2d(in_channels = outplanes,
                               out_channels = outplanes,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1)

        self.bn1 = nn.BatchNorm2d(outplanes)

        self.conv3 = nn.Conv2d(in_channels = inplanes,
                               out_channels = outplanes,
                               kernel_size = 1,
                               stride = stride,
                               padding = 0)

        self.bn2 = nn.BatchNorm2d(outplanes)

        self.se1 = SE(channels = outplanes,
                      ratio = 8)

        self.out = nn.Conv2d(in_channels = 2 * outplanes,
                               out_channels = outplanes,
                               kernel_size = 1,
                               stride = stride,
                               padding = 0)



    def forward(self,x):

        x1 = x #
        x = self.conv1(x)
        x = self.bn1(self.conv2(x))
        x1 = self.bn2(self.conv3(x1))

        x1 = self.se1(x1)

        out = torch.cat((x1,x),dim =1)

        out = self.out(out)

        #out = self.relu(x + x1)

        return  out


class DF(nn.Module):

    def __init__(self, inplanes = 64,scale_factor = 4):
        super(DF, self).__init__()

        self.relu = nn.LeakyReLU(inplace = True)

        # 上采样系数
        self.scale_factor = scale_factor

        self.Up = nn.Upsample(scale_factor = self.scale_factor, mode = 'bilinear', align_corners = True)

        number_f = inplanes * 2

        #   FLW-Net
        self.e_conv1 = CSDN_Tem(inplanes, number_f)
        self.e_conv2 = CSDN_Tem(number_f, number_f)
        self.e_conv3 = CSDN_Tem(number_f + inplanes, number_f)
        self.e_conv4 = CSDN_Temd(number_f, inplanes)

        #   TODO?
        self.g_conv2 = Hist_adjust(inplanes, inplanes)


        self.fc = LowFR(inplanes = inplanes,
                        ksize = 3,stride = 1)








    def forward(self, x):
        # source domain    target domain cdf
        # encoder
        xd = F.interpolate(x, scale_factor = 1. / self.scale_factor, mode = 'bilinear',
                           align_corners = True)

        x1 = x.clone()

        out = self.e_conv2(self.e_conv1(x1))
        #noise = torch.randn_like(x)
        # global
        xd = self.Up(xd)

        out = torch.cat((out, xd), dim = 1)

        out1 = self.e_conv4(self.e_conv3(out))

        recon = self.g_conv2(out1)

        # decoder
        out = self.fc(recon)


        return out


if __name__ == '__main__':
    x = torch.randn((1,3,256,256))

    model = DF()

    out = model(x)

    print(out.shape)