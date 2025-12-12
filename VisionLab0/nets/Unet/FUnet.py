# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import sys,os
# sys.path.insert(0,os.getcwd())
# # from nets.Transim.sample import StyleTransfer,kl_loss,mse_loss,enhance_net_nopool
# from nets.FTMNet import MixFreFeature
# from nets.backbone import CBA


# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.double_conv = nn.Sequential(
#                             CBA(c = in_channels,outc = out_channels,k = 3,p=1),
#                             CBA(c = out_channels, outc = out_channels, k = 3, p = 1),
#         )

#     def forward(self, x):
#         return self.double_conv(x)

# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)

# class Up(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

#         self.conv = DoubleConv(in_channels, out_channels)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         #input is CHW
#         diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
#         diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])

#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)


# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)


# class FUNet(nn.Module):
#     def __init__(self, n_channels = 3, n_classes = 2, bilinear=True):
#         super(FUNet, self).__init__()

#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         self.down4 = Down(512, 1024)
#         self.up1 = Up(1536, 512, bilinear)

#         self.freq1 = MixFreFeature(128)
#         self.freq2 = MixFreFeature(256)
#         self.up2 = Up(768, 256, bilinear)
#         self.up3 = Up(384, 128, bilinear)
#         self.up4 = Up(192, 64, bilinear)
#         self.outc = OutConv(64, n_classes)

#         self.boundary = OutConv(64,1)
#         self.points = OutConv(128,1)

#     def forward(self, x):


#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x2 = self.freq1(x2)
#         x3 = self.down2(x2)
#         x3 = self.freq2(x3)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)

#         x = self.up2(x, x3)
#         x = self.up3(x, x2)


#         points = self.points(x)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         boundary = self.boundary(x)

#         return logits,boundary,points


#     def _forward(self,x):

#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)

#         logits = self.outc(x)

#         return logits

# if __name__ == '__main__':
#     x = torch.randn((1,3,256,256))
#     net = FUNet(n_channels=3, n_classes=2,bilinear = True)
#     out = net(x)

#     for o in out:
#      print(o.shape)

#     # from nets.USI import flops_parameters
#     #
#     # """
#     # FLOPs:55.989G
#     # Para: 31.385M
#     # FPS: 161.73
#     # """
#     # flops_parameters(net, input=(x,))
#     #
#     # # net.get_params()
#     #
#     # from utils.tools import get_fps
#     #
#     # # for i in range(5):
#     # #
#     # #  get_fps(net = net,input_size = (512,512),device=torch.device('cuda:0'))
#     # from datasets.dataloader import BuildDataset, collate_seg
#     # from torch.utils.data import DataLoader
#     #
#     # path = r'D:\JinKuang\RTSeg\_Aerial_val.txt'
#     # path1 = r'D:\JinKuang\RTSeg\_M_val.txt'
#     # data = BuildDataset(train_file=path1, augment=True)
#     # train_loader = DataLoader(data, batch_size=1, shuffle=False, collate_fn=collate_seg)
#     # train_iter = iter(train_loader)
#     # device = torch.device('cuda')
#     # fps = 0
#     # for i, batch in enumerate(train_loader):
#     #     image, label, png, (heat2, heat4), \
#     #         ((boundary1, boundary2), boundary4, boundary8, boundary16) = batch
#     #     image = image.to(device)
#     #
#     #     image = F.interpolate(image,size=(256,256),mode='bilinear',align_corners=True)
#     #     fps = max(fps, get_fps(net=net, input=image, device=device))
#     #
#     #     print(f'Max_fps: {fps}')



import torch
import torch.nn as nn
import torch.nn.functional as F
import sys,os


sys.path.insert(0,os.getcwd())

from nets.FTMNet import MixFreFeature,Attention
from nets.Transim.network import LowFR,DF


from nets.backbone import CBA
from sklearn.cluster import KMeans

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
class KMeansClustering(nn.Module):
    def __init__(self, num_clusters, feature_dim):
        super(KMeansClustering, self).__init__()
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim

    def fit(self, features):
        """
        使用K-means对输入特征进行聚类，并返回聚类中心（原型）。
        features: Tensor，形状为 [batch_size, height*width, feature_dim]
        """
        features_np = features.detach().cpu().numpy()  # 转换为NumPy数组
        kmeans = KMeans(n_clusters = self.num_clusters)
        kmeans.fit(features_np.reshape(-1, self.feature_dim))  # [batch_size*height*width, feature_dim]

        # 获取聚类中心
        cluster_centers = torch.tensor(kmeans.cluster_centers_).float().cuda(features.device)
        return cluster_centers

    def forward(self, features):
        """
        将输入特征映射到最接近的聚类中心。
        """
        features_flattened = features.view(features.size(0), -1,
                                           features.size(-1))  # 展平成 [batch_size, height*width, feature_dim]

        # 对特征进行K-means聚类
        cluster_centers = self.fit(features_flattened)

        # 计算每个特征与聚类中心的距离
        distances = torch.cdist(features_flattened, cluster_centers)  # [batch_size, height*width, num_clusters]

        # 找到每个特征最接近的聚类中心
        cluster_assignments = torch.argmin(distances, dim = -1)  # [batch_size, height*width]

        return cluster_assignments, cluster_centers

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class FUNet(nn.Module):
    def __init__(self, n_channels = 3, n_classes = 2, bilinear=True):
        super(FUNet, self).__init__()

        self.net = DF(inplanes = 64)

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.conv = CBA(k = 3,
                        c = 1024 + 64,outc = 1024)
        self.up1 = Up(1536, 512, bilinear)

        self.freq1 = MixFreFeature(128)
        self.freq2 = MixFreFeature(256)
        self.df1 = LowFR(128)
        # self.df2 = LowFR(256)
        self.up2 = Up(768, 256, bilinear)
        self.up3 = Up(384, 128, bilinear)
        self.up4 = Up(192, 64, bilinear)
        self.df3 = LowFR(256)
        self.outc = OutConv(64, n_classes)
         # K-means聚类模块
        #self.kmeans1 = KMeansClustering(n_classes, 512)


        self.boundary = OutConv(64,1)
        self.points = OutConv(128,1)

    def apply_kmeans(self, kmeans,features):
        """
        对特征图应用K-means聚类，生成原型（聚类中心），并将特征图映射到原型空间。
        features: Tensor，形状为 [batch_size, channels, height, width]
        """
        # 将特征图展平成 [batch_size, height*width, channels] 形状
        batch_size, channels, height, width = features.size()
        features_flattened = features.view(batch_size, channels, -1).transpose(1,
                                                                               2)  # [batch_size, height*width, channels]

        # 使用K-means进行聚类，获取聚类标签和聚类中心（原型）
        cluster_assignments, cluster_centers = kmeans(features_flattened)

        # 返回经过K-means聚类后的特征图和聚类中心
        return features, cluster_assignments, cluster_centers

    def forward(self, x):

        


        x1 = self.inc(x)

        out = self.net(x1)

        x2 = self.down1(x1)
        x2 = self.freq1(x2)
        x2 = self.df1(x2)
        x3 = self.down2(x2)
        x3 = self.freq2(x3)
       # x3 = self.df2(out1)


        x4 = self.down3(x3)
        x5 = self.down4(x4)

        

        b,c,h,w = x5.shape
        out = F.interpolate(out,(h,w),mode = 'bilinear',
                            align_corners = True)
        out = torch.cat((out,x5),dim = 1)
        x5 = self.conv(out)
        # x5 = x5 + out

        x = self.up1(x5, x4)
        #x,*_ = self.apply_kmeans(self.kmeans1,x) 

        x = self.up2(x, x3)
        x = self.df3(x)

        x = self.up3(x, x2)

        points = self.points(x)
        x = self.up4(x, x1)
        logits = self.outc(x)
        boundary = self.boundary(x)

        return logits,boundary,points


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
    net = FUNet(n_channels=3, n_classes=2,bilinear = True)
    out = net(x)

    for o in out:
     print(o.shape)
    torch.save(net.state_dict(),'-1.pth')

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
