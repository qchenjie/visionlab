import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


class UNetWithPrototypes(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetWithPrototypes, self).__init__()
        self.num_clusters = out_channels  # 聚类的数量

        # 编码器部分
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # 解码器部分
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)

        # 最终输出层
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size = 1)

        # K-means聚类模块
        self.kmeans1 = KMeansClustering(self.num_clusters, 512)
        # self.kmeans2 = KMeansClustering(self.num_clusters, 256)
        # self.kmeans3 = KMeansClustering(self.num_clusters, 128)
        # self.kmeans4 = KMeansClustering(self.num_clusters, 64)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        # 编码阶段
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # 通过K-means聚类进行原型构建（对每个特征图进行聚类）
        enc4, cluster_assignments_4, cluster_centers_4 = self.apply_kmeans(self.kmeans1,enc4)

        # 解码阶段

        dec3 = self.dec3(
            torch.cat([ enc3, F.interpolate(enc4, enc3.size()[ 2: ], mode = 'bilinear', align_corners = True) ], 1))
        dec2 = self.dec2(
            torch.cat([ enc2, F.interpolate(dec3, enc2.size()[ 2: ], mode = 'bilinear', align_corners = True) ], 1))
        dec1 = self.dec1(
            torch.cat([ enc1, F.interpolate(dec2, enc1.size()[ 2: ], mode = 'bilinear', align_corners = True) ], 1))

        # 最终输出
        out = self.final_conv(dec1)

        return out

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


if __name__ == '__main__':
    x = torch.randn((1,3,256,256)).cuda()

    model = UNetWithPrototypes(in_channels = 3,
                               out_channels = 2).cuda()

    out = model(x)
    print(out.shape)
