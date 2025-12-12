# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import numpy as np
import torch
from torch import nn
from nets.GCN.torch_nn import BasicConv, batched_index_select, act_layer
from nets.GCN.torch_edge import DenseDilatedKnnGraph
from nets.GCN.pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.models.layers import DropPath


class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        return self.nn(x)


class MRAConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRAConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)
        self.attention = nn.Sequential(
            nn.Linear(12, 1),  
            nn.Softmax(dim=-1) 
        )

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        # print("xi",x_i.shape)
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        a_weight= self.attention(x_j-x_i)
        # print(a_weight.shape)
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)     
        x_j=torch.mul(a_weight, x_j)
        # print("xj",x_j.shape)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)

        return self.nn(x)
    
class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1))


class GINConv2d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j = torch.sum(x_j, -1, keepdim=True)
        return self.nn((1 + self.eps) * x + x_j)




#-------------------------------------------------#
#   TODO?
#-------------------------------------------------#
class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRAConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)

def print_patch_connections(edge_index):
    num_patches = edge_index.size(2)
    connections = {}
    for i in range(num_patches):
        query_id = edge_index[0, 0, i, 0].item()
        connected_nodes = []
        for j in range(edge_index.size(3)):
            if edge_index[0, 0, i, j] != -1:
                connected_nodes.append(edge_index[1, 0, i, j].item())
        connections[query_id] = connected_nodes

    for query_id, connected_nodes in connections.items():
        print(f"patch {query_id}: {connected_nodes}")



class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()            
        x = x.reshape(B, C, -1, 1).contiguous()

        # 获取邻接信息
        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        x = super(DyGraphConv2d, self).forward(x, edge_index, y)
        return x.reshape(B, -1, H, W).contiguous()


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, r)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            print('using relative_pos')
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        B, C, H, W = x.shape
        # TODO? get vert
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        x = self.graph_conv(x, relative_pos)
        # TODO? res
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
        return x


class UP(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(UP,self).__init__()


        self.layer = nn.Sequential(
                     nn.Upsample(scale_factor = 2,mode = 'bilinear',align_corners = True),
                     nn.Conv2d(in_channels = inchannel,
                            out_channels = inchannel, kernel_size = 3, stride = 1, padding = 1),
                     nn.BatchNorm2d(inchannel),
                     nn.LeakyReLU(inplace = True)
        )

        self.conv =  nn.Conv2d(in_channels = inchannel + outchannel,
                            out_channels = outchannel, kernel_size = 3, stride = 1, padding = 1)
    def forward(self,x,x1):

        x = self.layer(x)
        x = torch.cat((x,x1),dim = 1)
        return self.conv(x)



class Down(nn.Module):
    def __init__(self,inchannel,
                 outchannel):
        super(Down, self).__init__()
        self.layer = nn.Sequential(
                     nn.MaxPool2d(stride = 2,padding = 1,kernel_size = 3),
                     nn.Conv2d(in_channels = inchannel,
                               out_channels = outchannel,kernel_size = 3,stride = 1,padding = 1),
                     nn.Conv2d(in_channels = outchannel,
                               out_channels = outchannel,kernel_size = 3,stride = 1,padding = 1),
                     nn.BatchNorm2d(outchannel),
                     nn.LeakyReLU(inplace = True)
        )

    def forward(self,x):

        x = self.layer(x)
        return x







class GraphSeg(nn.Module):
    def __init__(self,inchannel,hidden,num_classes):
        super(GraphSeg, self).__init__()

        self.hidden = hidden
        self.num_class = num_classes
        self.conv1 = nn.Conv2d(in_channels = inchannel,out_channels = hidden,
                               kernel_size = 3,stride = 1,padding = 1)


        self.layer1 = nn.Sequential(
                      Grapher(in_channels = hidden),
                      Down(inchannel = hidden,outchannel = hidden * 2)
        )

        self.layer2 = nn.Sequential(
                      Grapher(in_channels = hidden * 2),
                      Down(inchannel = hidden * 2,outchannel = hidden * 4)
        )

        self.layer3 = nn.Sequential(
            Grapher(in_channels = hidden * 4),
            Down(inchannel = hidden * 4, outchannel = hidden * 8)
        )

        self.layer4 = nn.Sequential(
            Grapher(in_channels = hidden * 8),
            Down(inchannel = hidden * 8, outchannel = hidden * 16)
        )

        self.layer5 = nn.Sequential(
            Grapher(in_channels = hidden * 16),
            Down(inchannel = hidden * 16, outchannel = hidden * 32)
        )


        self.up1 = UP(hidden * 32,hidden * 16)
        self.up2 = UP(hidden * 16,hidden * 8)
        self.up3 = UP(hidden * 8,hidden * 4)
        self.up4 = UP(hidden * 4,hidden * 2)
        self.up5 = UP(hidden * 2,hidden)

        self.out = nn.Sequential(nn.Conv2d(hidden,hidden,kernel_size = 3,
                                             stride = 1,padding = 1),
                                   nn.LeakyReLU(inplace = True),
                                     nn.Conv2d(hidden, num_classes, kernel_size = 1,
                                             stride = 1)
                                   )





    def forward(self,x):

        x = self.conv1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)



        up1 = self.up1(x5,x4)
        up2 = self.up2(up1,x3)
        up3 = self.up3(up2,x2)
        up4 = self.up4(up3,x1)
        up5 = self.up5(up4,x)


        return self.out(up5)






if __name__ == '__main__':

    # 对图网络进行采样
    x = torch.randn((1,3,224,224))

    model = GraphSeg(inchannel = 3,hidden = 64,num_classes = 2)

    out = model(x)
    #print(out)

    print(out.shape)