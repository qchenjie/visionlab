# #!/usr/bin/python
# # -*- encoding: utf-8 -*-
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# from nets.USI import SCOT,CBA,ResCBA
# from nets._sta_ import Transformer
#
# BatchNorm2d = nn.BatchNorm2d
#
#
# # -----------------------------------------------------------#
# #
# # -----------------------------------------------------------#
# class ContextPath(nn.Module):
#     def __init__(self,  pretrain_model = '', use_conv_last = False, *args, **kwargs):
#         super(ContextPath, self).__init__()
#
#
#         self.backbone = SCOT( use_conv_last = use_conv_last)
#
#
#         #-----------------------------------------------------#
#         #
#         #-----------------------------------------------------#
#         inplanes = 256
#
#         self.arm16 = FeatureFusionMatch(inplanes, inplanes//2,128)
#
#         self.arm32 = FeatureFusionMatch(inplanes * 2, inplanes,128)
#
#         self.conv_avg = CBA(inplanes * 2, 128,  1,  1)
#
#
#
#
#     def forward(self, x):
#         H0, W0 = x.size()[ 2: ]
#
#         feat2, feat4, feat8, feat16, feat32 = self.backbone(x)
#         H8, W8 = feat8.size()[ 2: ]
#         H16, W16 = feat16.size()[ 2: ]
#         H32, W32 = feat32.size()[ 2: ]
#
#         avg = F.upsample(feat32, feat16.size()[ 2: ],mode = 'bilinear',align_corners = True)
#
#         avg_up = self.conv_avg(avg)# 128 32 32
#         #avg_up = F.interpolate(avg, (H32, W32), mode = 'nearest')
#
#
#         feat32_arm = self.arm32(feat16,feat32) # 128 32 32
#         feat32_up = feat32_arm + avg_up
#
#
#         feat16_arm = self.arm16(feat8,feat16)
#         feat16_up = feat16_arm + F.interpolate(feat32_up,feat16_arm.size()[2:],
#                                                mode = 'bilinear',align_corners = True)
#
#         # 2 4 8 16 8 16
#         return feat2, feat4, feat8, feat16, feat16_up, feat32_up  # x8, x16
#
# #---------------------------------------------------------#
# #
# #---------------------------------------------------------#
# class FeatureFusionMatch(nn.Module):
#     def __init__(self, features1, feature2,outc):
#         super(FeatureFusionMatch, self).__init__()
#
#         self.delta_gen1 = nn.Sequential(
#
#             nn.Conv2d(2 * outc, 2, kernel_size = 3, padding = 1, bias = False)
#         )
#
#         self.delta_gen2 = nn.Sequential(
#             nn.Conv2d(2 * outc, 2, kernel_size = 3, padding = 1, bias = False)
#         )
#
#         self.conv = nn.Conv2d(in_channels = features1, out_channels = outc,
#                               kernel_size = 3,padding = 1,bias = False)
#         self.conv1 = nn.Conv2d(in_channels = feature2, out_channels = outc,
#                               kernel_size = 3,padding = 1, bias = False)
#
#         self.delta_gen1[ -1 ].weight.data.zero_()
#         self.delta_gen2[ -1 ].weight.data.zero_()
#
#     # TODO： reference https://github.com/speedinghzl/AlignSeg/issues/7
#     def bilinear_interpolate_torch_gridsample(self, input, size, delta = 0):
#         out_h, out_w = size  # 输出的尺度
#         n, c, h, w = input.shape  # 输入的尺度
#         s = 2.0  # 缩放比例
#
#         # 归一化
#         norm = torch.tensor([ [ [ [ (out_w - 1) / s, (out_h - 1) / s ] ] ] ]).type_as(input).to(
#             input.device)  # not [h/s, w/s]
#
#         # TODO offset
#         # 值域范围
#         w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
#         h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
#
#         #
#         grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
#         grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
#         grid = grid + delta.permute(0, 2, 3, 1) / norm
#
#         # 采样点
#         output = F.grid_sample(input, grid, align_corners = True)
#         return output
#
#     def forward(self, low_stage, high_stage):
#         #---------------------------------------------------------#
#         #      浅层 深层
#         #---------------------------------------------------------#
#         h, w = low_stage.size(2), low_stage.size(3)
#         high_stage = F.interpolate(input = high_stage, size = (h, w), mode = 'bilinear', align_corners = True)
#         high_stage = self.conv(high_stage)
#         low_stage = self.conv1(low_stage)
#
#         concat = torch.cat((low_stage, high_stage), 1)
#
#         delta1 = self.delta_gen1(concat)
#         delta2 = self.delta_gen2(concat)
#
#         high_stage = self.bilinear_interpolate_torch_gridsample(high_stage, (h, w), delta1)
#         low_stage = self.bilinear_interpolate_torch_gridsample(low_stage, (h, w), delta2)
#
#         high_stage += low_stage  # 这里作为直接
#         return high_stage
#
#
#
# # -----------------------------------------------------------#
# #
# # -----------------------------------------------------------#
# class FeatureFusionModule(nn.Module):
#     def __init__(self, in_chan, out_chan, *args, **kwargs):
#         super(FeatureFusionModule, self).__init__()
#         self.convblk = ResCBA(in_chan, out_chan, 1, stride = 1)
#         self.conv1 = nn.Conv2d(out_chan,
#                                out_chan // 4,
#                                kernel_size = 1,
#                                stride = 1,
#                                padding = 0,
#                                bias = False)
#         self.conv2 = nn.Conv2d(out_chan // 4,
#                                out_chan,
#                                kernel_size = 1,
#                                stride = 1,
#                                padding = 0,
#                                bias = False)
#         self.relu = nn.ReLU(inplace = True)
#         self.sigmoid = nn.Sigmoid()
#
#
#     def forward(self, fsp, fcp):
#         fcat = torch.cat([ fsp, fcp ], dim = 1)
#         feat = self.convblk(fcat)
#         atten = F.avg_pool2d(feat, feat.size()[ 2: ])
#         atten = self.conv1(atten)
#         atten = self.relu(atten)
#         atten = self.conv2(atten)
#
#         atten = self.sigmoid(atten)
#         feat_atten = torch.mul(feat, atten)
#         feat_out = feat_atten + feat
#         return feat_out
#
#
#
# class BoundaryHead(nn.Module):
#     def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
#         super(BoundaryHead, self).__init__()
#         self.conv = CBA(in_chan, mid_chan, 3, 1)
#         self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.conv_out(x)
#         return x
#
# #-------------------------------------------------------------#
# #
# #-------------------------------------------------------------#
# class BuildRT(nn.Module):
#     def __init__(self, n_classes,  use_conv_last = False,*args,
#                  **kwargs):
#         super(BuildRT, self).__init__()
#
#
#         # 上下文注意力模块
#         self.cp = ContextPath( use_conv_last = use_conv_last)
#
#         # 下分支
#         # self.lln = Transformer(heads = 32,dim = 8,
#         #                        depth = 1,mlp_dim = 16)
#
#         conv_out_inplanes = 128
#         sp2_inplanes = 16 #2
#         sp4_inplanes = 32 #4
#         sp8_inplanes = 128 #8
#         sp16_inplanes = 256 #16
#         inplane = sp8_inplanes + conv_out_inplanes
#
#
#
#         # 特征聚合模块 256 -> 128
#         self.ffm = FeatureFusionModule(inplane, 128)
#
#         # self.linear = nn.Sequential(
#         #               nn.LayerNorm(32),
#         #               nn.Linear(32,8),
#         #               nn.Dropout(0.5)
#         # )
#
#         #----------------------------------------------------------------#
#         #                          分割头
#         #----------------------------------------------------------------#
#         self.conv_out = BoundaryHead(128, 64, n_classes)
#         self.conv_out16 = BoundaryHead(conv_out_inplanes, 64, n_classes)
#         self.conv_out32 = BoundaryHead(conv_out_inplanes, 64, n_classes)
#
#         #--------------------------------------------------------#
#         #                   边界损失
#         #--------------------------------------------------------#
#         self.conv_out_sp16 = BoundaryHead(sp16_inplanes , 64, 1)
#         self.conv_out_sp8 = BoundaryHead(sp8_inplanes , 64, 1)
#         self.conv_out_sp4 = BoundaryHead(sp4_inplanes + 1, 64, 1)
#         self.conv_out_sp2 = BoundaryHead(sp2_inplanes + 1, 64, 1)
#
#         #--------------------------------------------------------#
#         #                   loc loss
#         #--------------------------------------------------------#
#         self.loc4 = BoundaryHead(sp4_inplanes, 32, 1)
#         self.loc2 = BoundaryHead(sp2_inplanes,32,1)
#
#     def forward(self, x,*boundary):
#         H, W = x.size()[ 2: ]
#         # 2 4 8 16 8 16
#         feat_res2, feat_res4, feat_res8, feat_res16, feat_cp8, feat_cp16 = self.cp(x)
#
#
#         # lln_featres2 = feat_res4
#         # b,c,h,w = feat_res4.shape
#         # lln_featres2 = lln_featres2.permute((0,2,3,1)).view((b,-1,c))
#         # lln_featres2 = self.linear(lln_featres2)
#         # lln_featres2 = self.lln(lln_featres2)
#
#
#         #-----------------------------------------------#
#         #                     边界损失
#         #-----------------------------------------------#
#         feat_out_sp2 = self.conv_out_sp2(torch.cat((feat_res2,boundary[0]),dim=1))
#
#         feat_out_sp4 = self.conv_out_sp4(torch.cat((feat_res4,boundary[1]),dim=1))
#
#         feat_out_sp8 = self.conv_out_sp8(feat_res8)
#
#         feat_out_sp16 = self.conv_out_sp16(feat_res16)
#
#         #------------------------------------------------#
#         #                   定位损失
#         #------------------------------------------------#
#         loc2 = self.loc2(feat_res2)
#         loc4 = self.loc4(feat_res4)
#
#
#
#         #-----------------------------------------------#
#         #           使用 8 times： feature fusion
#         #-----------------------------------------------#
#         feat_fuse = self.ffm(feat_res8, feat_cp8)
#
#         feat_out = self.conv_out(feat_fuse)
#         feat_out16 = self.conv_out16(feat_cp8)
#         feat_out32 = self.conv_out32(feat_cp16)
#
#         # lln_featres2 = lln_featres2.view((b,h,w,c)).permute((0,3,1,2))
#         # lln_featres2 = F.interpolate(lln_featres2, (H, W), mode = 'bilinear', align_corners = True)
#         feat_out8 = F.interpolate(feat_out, (H, W), mode = 'bilinear', align_corners = True)
#         feat_out16 = F.interpolate(feat_out16, (H, W), mode = 'bilinear', align_corners = True)
#         feat_out32 = F.interpolate(feat_out32, (H, W), mode = 'bilinear', align_corners = True)
#
#         return (feat_out8,feat_out16,feat_out32),\
#                (feat_out_sp2,feat_out_sp4,feat_out_sp8,feat_out_sp16),\
#                (loc2,loc4) #down
#
#
#
# def flops_parameters(model = None,
#                      input = None,
#                      verbose = False):
#     from thop import profile
#     flops, para = profile(model, input, verbose = verbose)
#
#     f = '{:.3f}G'.format(flops / (1e9))
#     p = '{:.3f}M'.format(para / (1e6))
#
#     print(f'FLOPs: {f}')
#     print(f'Para: {p}')
#     return f, p
#
#
# if __name__ == "__main__":
#     net = BuildRT(n_classes = 2)
#     net.cuda()
#     net.eval()
#     in_ten = torch.randn(1, 3, 256, 256).cuda()
#     b1 = torch.randn(1, 1, 128, 128).cuda()
#     b2 = torch.randn(1, 1, 64, 64).cuda()
#
#     out, out16, out32 = net(in_ten,*(b1,b2))
#
#     for o in out:
#         print(o.shape)
#     print('+++++++++++++++++++++++++')
#
#     for o in out16:
#         print(o.shape)
#     print('+++++++++++++++++++++++++')
#     for o in out32:
#         print(o.shape)
#
#     #print(out.shape, out16.shape, out32.shape)
#
#     # 计算模型的参数量
#     flops_parameters(model = net, input = (in_ten,*(b1,b2)))
#
#     # from torchsummary import summary
#     #
#     # summary(net,(3,512,512))
#
#     #torch.save(net.state_dict(), 'BuildRT.pth')
#
#
#     a = torch.randn((3,3))
#     b = torch.randn((3,3))
#
#     diff = (a - b)
#
#     print(diff)
#
#
import time

#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from nets.USI import SCOT,CBA,ResCBA
from nets._sta_ import Transformer

BatchNorm2d = nn.BatchNorm2d




# -----------------------------------------------------------#
#
# -----------------------------------------------------------#
class ContextPath(nn.Module):
    def __init__(self,  pretrain_model = '', use_conv_last = False, *args, **kwargs):
        super(ContextPath, self).__init__()


        self.backbone = SCOT( use_conv_last = use_conv_last)


        #-----------------------------------------------------#
        #
        #-----------------------------------------------------#
        inplanes = 256


        # TODO
        self.arm16 = FeatureFusionMatch(inplanes, inplanes//2,inplanes//2)


        self.arm8 = FeatureFusionMatch(inplanes //2, inplanes//8,inplanes//2)

        self.conv_avg = CBA(inplanes , inplanes//2,  1,  1)

        self.out = CBA(inplanes//2,32)




    def forward(self, x):
        #H0, W0 = x.size()[ 2: ]

        feat2, feat4, feat8, feat16 = self.backbone(x)


        # H8, W8 = feat8.size()[ 2: ]
        # H16, W16 = feat16.size()[ 2: ]
       # H32, W32 = feat32.size()[ 2: ]

        avg = F.upsample(feat16, feat8.size()[ 2: ],mode = 'bilinear',align_corners = True)

        avg_up = self.conv_avg(avg)# 128 32 32
        #avg_up = F.interpolate(avg, (H32, W32), mode = 'nearest')


        feat8_arm = self.arm16(feat8,feat16) # 128 32 32
        feat8_up = feat8_arm + avg_up


        feat4_arm = self.arm8(feat4,feat8)

        # 128 -> 32
        feat4_up = feat4_arm + F.interpolate(feat8_up,feat4_arm.size()[2:],
                                               mode = 'bilinear',align_corners = True)
        feat4_up = self.out(feat4_up)
        # 2 4 8 16 8 16
        return feat2, feat4, feat8, feat16, feat4_up, feat8_up  # x8, x16

#---------------------------------------------------------#
#
#---------------------------------------------------------#
class FeatureFusionMatch(nn.Module):
    def __init__(self, features1, feature2,outc):
        super(FeatureFusionMatch, self).__init__()

        self.delta_gen1 = nn.Sequential(

            nn.Conv2d(2 * outc, 2, kernel_size = 3, padding = 1, bias = False)
        )

        self.delta_gen2 = nn.Sequential(
            nn.Conv2d(2 * outc, 2, kernel_size = 3, padding = 1, bias = False)
        )

        self.conv = nn.Conv2d(in_channels = features1, out_channels = outc,
                              kernel_size = 3,padding = 1,bias = False)
        self.conv1 = nn.Conv2d(in_channels = feature2, out_channels = outc,
                              kernel_size = 3,padding = 1, bias = False)

        self.delta_gen1[ -1 ].weight.data.zero_()
        self.delta_gen2[ -1 ].weight.data.zero_()

    # TODO： reference https://github.com/speedinghzl/AlignSeg/issues/7
    def bilinear_interpolate_torch_gridsample(self, input, size, delta = 0):
        out_h, out_w = size  # 输出的尺度
        n, c, h, w = input.shape  # 输入的尺度
        s = 2.0  # 缩放比例

        # 归一化
        norm = torch.tensor([ [ [ [ (out_w - 1) / s, (out_h - 1) / s ] ] ] ]).type_as(input).to(
            input.device)  # not [h/s, w/s]

        # TODO offset
        # 值域范围
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)

        #
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        # 采样点
        output = F.grid_sample(input, grid, align_corners = True)
        return output

    def forward(self, low_stage, high_stage):
        #---------------------------------------------------------#
        #      浅层 深层
        #---------------------------------------------------------#
        h, w = low_stage.size(2), low_stage.size(3)
        high_stage = F.interpolate(input = high_stage, size = (h, w), mode = 'bilinear', align_corners = True)
        high_stage = self.conv(high_stage)
        low_stage = self.conv1(low_stage)

        concat = torch.cat((low_stage, high_stage), 1)

        delta1 = self.delta_gen1(concat)
        delta2 = self.delta_gen2(concat)

        high_stage = self.bilinear_interpolate_torch_gridsample(high_stage, (h, w), delta1)
        low_stage = self.bilinear_interpolate_torch_gridsample(low_stage, (h, w), delta2)

        high_stage += low_stage  # 这里作为直接
        return high_stage



# -----------------------------------------------------------#
#
# -----------------------------------------------------------#
class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ResCBA(in_chan, out_chan, 1, stride = 1)
        self.conv1 = nn.Conv2d(out_chan,
                               out_chan // 4,
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               bias = False)
        self.conv2 = nn.Conv2d(out_chan // 4,
                               out_chan,
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               bias = False)
        self.relu = nn.ReLU(inplace = True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, fsp, fcp):
        fcat = torch.cat([ fsp, fcp ], dim = 1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[ 2: ])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)

        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out



class BoundaryHead(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BoundaryHead, self).__init__()
        self.conv = CBA(in_chan, mid_chan, 3, 1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

#-------------------------------------------------------------#
#
#-------------------------------------------------------------#
class BuildRT(nn.Module):
    def __init__(self, n_classes,  use_conv_last = False,*args,
                 **kwargs):
        super(BuildRT, self).__init__()

        self.p1 = nn.Parameter(torch.randn(1,256,256),requires_grad=True)

      #  self.p1 = self.p1[None,:,:,:]

        self.p2 = nn.Parameter(torch.randn(1,128,128),requires_grad=True)
        #self.p2 = self.p2[None, :, :, :]


        # 上下文注意力模块
        self.cp = ContextPath( use_conv_last = use_conv_last)

        # 下分支
        # self.lln = Transformer(heads = 32,dim = 8,
        #                        depth = 1,mlp_dim = 16)

        conv_out_inplanes = 32
        sp2_inplanes = 16 #2
        sp4_inplanes = 32 #4
        sp8_inplanes = 128 #8
        sp16_inplanes = 256 #16
        inplane = sp4_inplanes + conv_out_inplanes



        # 特征聚合模块 256 -> 128
        self.ffm = FeatureFusionModule(inplane , inplane // 2)

        # self.linear = nn.Sequential(
        #               nn.LayerNorm(32),
        #               nn.Linear(32,8),
        #               nn.Dropout(0.5)
        # )

        #----------------------------------------------------------------#
        #                          分割头
        #----------------------------------------------------------------#
        self.conv_out = BoundaryHead(inplane // 2, conv_out_inplanes // 4, n_classes)
        self.conv_out16 = BoundaryHead(conv_out_inplanes, conv_out_inplanes // 4, n_classes)
        self.conv_out32 = BoundaryHead(128, conv_out_inplanes // 4, n_classes)

        #--------------------------------------------------------#
        #                   边界损失
        #--------------------------------------------------------#
        self.conv_out_sp16 = BoundaryHead(sp16_inplanes , conv_out_inplanes // 2, 1)
        self.conv_out_sp8 = BoundaryHead(sp8_inplanes , conv_out_inplanes // 2, 1)
        self.conv_out_sp4 = BoundaryHead(sp4_inplanes , conv_out_inplanes // 2, 1)
        self.conv_out_sp2 = BoundaryHead(sp2_inplanes   , conv_out_inplanes // 2, 1)

        #--------------------------------------------------------#
        #                   loc loss
        #--------------------------------------------------------#
        self.loc4 = BoundaryHead(sp4_inplanes, 32, 1)
        self.loc2 = BoundaryHead(sp2_inplanes,32,1)

    def forward(self, x):#? TODO 删了


        H, W = x.size()[ 2: ]
        # 2 4 8 16 8 16
        # 2 4 8 16 4 8
        t0 = time.time()
        feat_res2, feat_res4, feat_res8, feat_res16, feat_cp8, feat_cp16 = self.cp(x)
        #print('第一个分支:',time.time() - t0)
        t1 = time.time()



        # lln_featres2 = feat_res4
        # b,c,h,w = feat_res4.shape
        # lln_featres2 = lln_featres2.permute((0,2,3,1)).view((b,-1,c))
        # lln_featres2 = self.linear(lln_featres2)
        # lln_featres2 = self.lln(lln_featres2)


        #-----------------------------------------------#
        #                     边界损失


        # with torch.no_grad():
        #     alpha1 = nn.MSELoss()(boundary[0],self.p1[None,:,:,:])
        # with torch.no_grad():
        #     alpha2 = nn.MSELoss()(boundary[1], self.p2[None, :, :, :])

        # feat_out_sp2 = alpha1  * self.conv_out_sp2(torch.cat((feat_res2,boundary[0] + self.p1[None,:,:,:]),dim=1))
        #
        # feat_out_sp4 = alpha2  * self.conv_out_sp4(torch.cat((feat_res4,boundary[1] + self.p2[None,:,:,:]),dim=1))

        # feat_out_sp2 = self.conv_out_sp2(feat_res2)
        #
        # feat_out_sp4 = self.conv_out_sp4(feat_res4)
        #
        #
        # feat_out_sp8 = self.conv_out_sp8(feat_res8)
        #
        # feat_out_sp16 = self.conv_out_sp16(feat_res16)
        #
        # #------------------------------------------------#
        # #                   定位损失
        # #------------------------------------------------#
        # loc2 = self.loc2(feat_res2)
        # loc4 = self.loc4(feat_res4)



        #-----------------------------------------------#
        #           使用 8 times： feature fusion
        #-----------------------------------------------#
        # feat_fuse = self.ffm(torch.cat((feat_res4,label),dim=1), feat_cp8)
        feat_fuse = self.ffm(feat_res4, feat_cp8)

        feat_out = self.conv_out(feat_fuse)

       # print('第二个模块:',time.time() - t1)
        feat_out16 = self.conv_out16(feat_cp8)
        # feat_out32 = self.conv_out32(feat_cp16)
        #
        # # lln_featres2 = lln_featres2.view((b,h,w,c)).permute((0,3,1,2))
        # # lln_featres2 = F.interpolate(lln_featres2, (H, W), mode = 'bilinear', align_corners = True)
        feat_out8 = F.interpolate(feat_out, (H, W), mode = 'bilinear', align_corners = True)
        # feat_out16 = F.interpolate(feat_out16, (H, W), mode = 'bilinear', align_corners = True)
        # feat_out32 = F.interpolate(feat_out32, (H, W), mode = 'bilinear', align_corners = True)
        #
        # return (feat_out8,feat_out16,feat_out32),\
        #        (feat_out_sp2,feat_out_sp4,feat_out_sp8,feat_out_sp16),\
        #        (loc2,loc4) #down
        return (feat_out8)



def flops_parameters(model = None,
                     input = None,
                     verbose = False):
    from thop import profile
    flops, para = profile(model, input, verbose = verbose)

    f = '{:.3f}G'.format(flops / (1e9))
    p = '{:.3f}M'.format(para / (1e6))

    print(f'FLOPs: {f}')
    print(f'Para: {p}')
    return f, p


if __name__ == "__main__":
    net = BuildRT(n_classes = 2)
    net.cuda()
    net.eval()
    in_ten = torch.randn(1, 3, 512, 512).cuda()
    b1 = torch.randn(1, 1, 256, 256).cuda()
    b2 = torch.randn(1, 1, 128, 128).cuda()

    label = torch.randn((1,2,128,128)).cuda()

    # out, out16, out32 = net(in_ten)#net(in_ten,label,*(b1,b2))
    #
    # for o in out:
    #     print(o.shape)
    # print('+++++++++++++++++++++++++')
    #
    # for o in out16:
    #     print(o.shape)
    # print('+++++++++++++++++++++++++')
    # for o in out32:
    #     print(o.shape)

    #print(out.shape, out16.shape, out32.shape)

    # 计算模型的参数量
    #flops_parameters(model = net, input = (in_ten,label,*(b1,b2),))

    # from torchsummary import summary
    #
    # summary(net,(3,512,512))

    #torch.save(net.state_dict(), 'BuildRT.pth')


    # a = torch.randn((3,3))
    # b = torch.randn((3,3))
    #
    # diff = (a - b)
    #
    # print(diff)
    x = torch.randn(1, 3, 512, 512).cuda()
    from nets.USI import flops_parameters

    """
    FLOPs: 14.086G
    Para: 2.397M
    FPS: 114.52
    """
    #flops_parameters(net, input=(x,))

    # outs = net(x)
    # for out in outs:
    #     print(out.size())

    from utils.tools import get_fps

    # for i in range(5):
    #
    #  get_fps(net = net,input_size = (512,512),device=torch.device('cuda:0'))
    from datasets.dataloader import BuildDataset, collate_seg
    from torch.utils.data import DataLoader

    path = r'D:\JinKuang\RTSeg\_Aerial_val.txt'
    path1 = r'D:\JinKuang\RTSeg\_M_val.txt'
    data = BuildDataset(train_file=path, augment=True)
    train_loader = DataLoader(data, batch_size=1, shuffle=True, collate_fn=collate_seg)
    train_iter = iter(train_loader)
    device = torch.device('cuda')
    fps = 0
    for i, batch in enumerate(train_loader):
        image, label, png, (heat2, heat4), \
            ((boundary1, boundary2), boundary4, boundary8, boundary16) = batch
        image = image.to(device)
        fps = max(fps, get_fps(net=net, input=image, device=device))

        print(f'Max_fps: {fps}')




