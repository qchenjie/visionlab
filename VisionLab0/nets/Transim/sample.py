import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.dataloader  import MedicalDataset, collate_seg
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from nets.Transim.ssim import SSIM
import torch.optim as  optimizer
import cv2

ssim_loss = SSIM()


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



    def forward(self,x):

        x1 = x #
        x = self.conv1(x)
        x = self.bn1(self.conv2(x))
        x1 = self.bn2(self.conv3(x1))

        x1 = self.se1(x1)

        out = self.relu(x + x1)

        return  out


class SE(nn.Module):
    def __init__(self,channels,ratio=16):
        super(SE, self).__init__()

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))

        self.fc=nn.Sequential(
            nn.Linear(channels,channels//ratio,False),
            nn.ReLU(),
            nn.Linear(channels//ratio, channels, False),
            nn.Sigmoid()
        )
    def forward(self,x):

        b,c,_,_ = x.size()

        avg = self.avgpool(x).view(b,c)

        y = self.fc(avg).view(b,c,1,1)
        return x * y.expand_as(x)  #TODO?


def pnsr(im1, im2):
    mse = np.mean((im1 - im2) ** 2)
    if mse == 0:
        return 100

    PIXEL_MAX = 255.

    return 20. * np.log10(PIXEL_MAX / np.sqrt(mse))


def ssim(y_true, y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    R = 255
    c1 = np.square(0.01 * R)
    c2 = np.square(0.03 * R)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)

    return ssim / denom


def mapping(source_img, target_img):
    # 计算直方图
    source_hist = cv2.calcHist([ source_img ], [ 0 ], None, [ 256 ], [ 0, 256 ])
    target_hist = cv2.calcHist([ target_img ], [ 0 ], None, [ 256 ], [ 0, 256 ])

    # 计算累积分布函数 TODO?
    source_cdf = np.cumsum(source_hist) / (np.sum(source_hist))
    target_cdf = np.cumsum(target_hist) / (np.sum(target_hist))

    mean = (target_cdf - np.min(source_cdf)) / (target_cdf - source_cdf + 1e-10)
    # 创建匹配映射

    target_cdf = np.where(source_cdf > mean, source_cdf, target_cdf)

    mapping = np.interp(source_cdf, target_cdf, range(256)).astype(np.uint8)

    # 应用匹配映射
    matched_img = mapping[ source_img ]
    return matched_img, target_cdf


def L_color(x):
    b, c, h, w = x.shape

    mean_rgb = torch.mean(x, [ 2, 3 ], keepdim = True)
    mr, mg, mb = torch.split(mean_rgb, 1, dim = 1)
    Drg = torch.pow(mr - mg, 2)
    Drb = torch.pow(mr - mb, 2)
    Dgb = torch.pow(mb - mg, 2)
    k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)

    return k


def L_retouch_mean(x, y):
    x = x.max(1, keepdim = True)
    y = y.max(1, keepdim = True)
    # L3_retouch_mean = torch.mean(torch.pow(x-torch.mean(y,[2,3],keepdim=True),2))
    L4_retouch_ssim = 1 - torch.mean(ssim_loss(x, y))

    return L4_retouch_ssim


def kl_loss(mean, std):
    loss = - 0.5 * torch.sum(1 + std - mean.pow(2) -
                             std.exp()) / mean.shape[ 0 ]

    return loss


def mse_loss(label, pred):
    B = label.shape[ 0 ]
    loss = nn.MSELoss(reduce = 'sum')

    l = loss(pred, label)

    return l


# 使用深度可分离卷积作为特征提取
# 轻量化
class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels = in_ch,
            out_channels = out_ch,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            groups = 1
        )
        # self.point_conv = nn.Conv2d(
        #     in_channels=in_ch,
        #     out_channels=out_ch,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     groups=1
        # )

    def forward(self, input):
        out = self.depth_conv(input)
        # out = nn.Tanh()(out)
        # out = self.point_conv(out)
        return out


class CSDN_Temd(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Temd, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels = in_ch,
            out_channels = out_ch,
            kernel_size = 3,
            stride = 1,
            padding = 2,  #
            dilation = 2,  # 还用空洞
            groups = 1  # DW
        )

    def forward(self, input):
        out = self.depth_conv(input)
        # out = self.point_conv(out)
        # out = nn.Tanh()(out)
        return out


# 调整V通道 使用卷积
class Hist_adjust(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Hist_adjust, self).__init__()
        self.point_conv = nn.Conv2d(
            in_channels = in_ch,
            out_channels = out_ch,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            groups = 1
        )

    def forward(self, input):
        out = self.point_conv(input)
        return out


def pool(x):
    b, c, h, w = x.shape
    poolx = F.adaptive_avg_pool2d(x, (1, w)).permute((0, 2, 3, 1)).contiguous().view(b, -1)
    pooly = F.adaptive_avg_pool2d(x, (h, 1)).permute((2, 3, 0, 1)).contiguous().view((b, -1))

    out = poolx  # + pooly

    return out


# # 模型网络构建
# class enhance_net_nopool(nn.Module):

#     def __init__(self, scale_factor = 8):
#         super(enhance_net_nopool, self).__init__()

#         self.relu = nn.LeakyReLU(inplace = True)

#         # 上采样系数
#         self.scale_factor = scale_factor

#         self.Up = nn.Upsample(scale_factor = self.scale_factor, mode = 'bilinear', align_corners = True)

#         number_f = 32

#         #   FLW-Net
#         self.e_conv1 = CSDN_Tem(3, number_f)
#         self.e_conv2 = CSDN_Tem(number_f, number_f)
#         self.e_conv3 = CSDN_Tem(number_f + 3, number_f)
#         self.e_conv4 = CSDN_Temd(number_f, 6)

#         #   GFE-Net
#         # self.g_conv1 = Hist_adjust(6, number_f)
#         self.g_conv2 = Hist_adjust(6, 3)
#         # self.g_conv3 = Hist_adjust(number_f, 3)

#         self.layer1 = nn.Sequential(nn.Linear(3 * 256 * 256, 1024),
#                                     nn.Dropout(0.5),
#                                     nn.Linear(1024, 256 * 3))

#         # self.layer2 = nn.Sequential(nn.Linear(3 * 128 * 128, 1024),
#         #                             nn.Dropout(0.5),
#         #                             nn.Linear(1024, 256 * 3))

#     def forward(self, x):
#         # source domain    target domain cdf

#         xd = F.interpolate(x, scale_factor = 1. / self.scale_factor, mode = 'bilinear',
#                            align_corners = True)

#         x1 = x.clone()

#         out = self.e_conv2(self.e_conv1(x1))
#         noise = torch.randn_like(x)
#         # global
#         xd = self.Up(xd)

#         out = torch.cat((out, xd), dim = 1)

#         out1 = self.e_conv4(self.e_conv3(out))
#         recon = nn.Sigmoid()(self.g_conv2(out1))

#         # 编码阶段
#         # out2 = self.g_conv2(self.g_conv1(out1)) #mean
#         # out3 = self.g_conv3(self.g_conv1(out1)) # var

#         # expout3 = torch.exp(0.5 * out3)
#         # noise1 = torch.randn_like(x)

#         # # TODO?
#         # recon = F.sigmoid(out2 + (expout3  * noise1))

#         # TODO?
#         b, c, h, w = recon.shape
#         out2 = recon.permute((0, 2, 3, 1)).contiguous().view(b, -1)
#         # out3 = out3.permute((0,2,3,1)).contiguous().view(b,-1)

#         recon = (self.layer1(out2).view(-1, 256, 3))

#         # out3 = (self.layer2(out3).view(-1, 256, 3))

#         return recon  # ,out2,out3



# 模型网络构建
class enhance_net_nopool(nn.Module):

    def __init__(self, scale_factor = 8):
        super(enhance_net_nopool, self).__init__()

        self.relu = nn.LeakyReLU(inplace = True)

        # 上采样系数
        self.scale_factor = scale_factor

        self.Up = nn.Upsample(scale_factor = self.scale_factor, mode = 'bilinear', align_corners = True)

        number_f = 16

        #   FLW-Net
        self.e_conv1 = CSDN_Tem(3, number_f)
        self.e_conv2 = CSDN_Tem(number_f, number_f)
        self.e_conv3 = CSDN_Tem(number_f + 3, number_f)
        self.e_conv4 = CSDN_Temd(number_f, 32)

        #   TODO?
        self.g_conv2 = Hist_adjust(32, 64)


        self.fc = LowFR(inplanes = 64,
                        ksize = 3,stride = 1)



        self.pool = nn.AdaptiveAvgPool2d((1,1))


        #-----------------------------------------------------------#
        #   TODO?
        #-----------------------------------------------------------#
        self.layer1 = nn.Sequential(nn.Linear(128,512),
                                    nn.Dropout(0.5),
                                    nn.Linear(512, 3*256))



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
        down1 = self.fc(recon)
        downproto = self.pool(down1)



        out2 = torch.cat((down1,downproto.expand_as(down1)),dim = 1)

        out2 = self.pool((out2))
        b,c,_,_ = out2.shape

        out2 = out2.view(b,-1)


        # out2 = self.g_conv2(self.g_conv1(out1)) #mean
        # out3 = self.g_conv3(self.g_conv1(out1)) # var

        # expout3 = torch.exp(0.5 * out3)
        # noise1 = torch.randn_like(x)

        # # TODO?
        # recon = F.sigmoid(out2 + (expout3  * noise1))

        # TODO?
        # b, c, h, w = recon.shape
        # out2 = recon.permute((0, 2, 3, 1)).contiguous().view(b, -1)
        # # out3 = out3.permute((0,2,3,1)).contiguous().view(b,-1)
        #
        # recon = (self.layer1(out2).view(-1, 256, 3))

        # out3 = (self.layer2(out3).view(-1, 256, 3))

        return self.layer1(out2).view(b,-1,3)






if __name__ == '__main__':
    init_iter = 0
    max_iter = 300
    lr = 2.5e-3

    source_file = 'WHUtarget.txt'
    target_file = 'S2source.txt'

    traindata = UnetDataset(source_file = source_file,
                            target_file = target_file)

    trainloader = DataLoader(batch_size = 8, dataset = traindata, collate_fn = collate_seg)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = enhance_net_nopool().to(device)

    # model.load_state_dict(torch.load('model_last.pth'))
    optim = optimizer.SGD(model.parameters(), lr = lr
                          , momentum = 0.95, weight_decay = 1e-3)

    for epoch in range(init_iter, max_iter):

        optim.zero_grad()

        for batch, (images, cdfs) in enumerate(trainloader):
            with torch.no_grad():
                images = images.float().to(device)

                # print(images.max(),images.min())

                cdfs = cdfs.float().to(device)

            recon = model(images)

            # loss1 = kl_loss(mean,std)
            loss2 = mse_loss(recon, cdfs)

            # loss3 = L_color(recon)
            # loss4 = L_retouch_mean(recon,cdfs)

            loss = (loss2)
            loss.backward()
            print(
                f'INFO==============ssim:{ssim(cdfs.detach().cpu().numpy(), recon.detach().cpu().numpy())}======pnsr:{pnsr(cdfs.detach().cpu().numpy(), recon.detach().cpu().numpy())}===loss:{loss.item()}================')

            optim.step()

    torch.save(model.state_dict(), f'model_last_{max_iter}.pth')

# if __name__ == '__main__':
#     model = enhance_net_nopool(scale_factor = 16)
#
#     # TODO same
#     x = torch.randn((1, 3, 512, 512))
#     hist = torch.randn(256,3)
#
#     output = model(x, hist)
#
#     for o in output:
#         print(o.shape)

# img,cdf = TargetCDF('../t1.jpg','../t2.jpg')
#
# print(img.shape,cdf.shape)
