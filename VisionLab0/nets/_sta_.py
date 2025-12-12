import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


def flops_parameters(model = None,
                     input = None,
                     verbose = False):

    from thop import profile
    flops,para = profile(model,input,verbose = verbose)

    f = '{:.3f}G'.format(flops / (1e9))
    p = '{:.3f}M'.format(para / (1e6))

    print(f'FLOPs: {f}')
    print(f'Para: {p}')
    return f,p



#FFN 前反馈网络
class FFN(nn.Module):
    def __init__(self,
                 dim,
                 hidden_dim,
                 dropout = 0.5):
        super(FFN, self).__init__()
        self.ffn = nn.Sequential(
                   nn.LayerNorm(dim),
                   nn.Linear(dim,hidden_dim),
                   nn.GELU(),
                   nn.Dropout(dropout), #抑制神经元
                   nn.Linear(hidden_dim,dim),
                   nn.Dropout(dropout)
        )

    def forward(self,x):

        return self.ffn(x)



# 自注意力机制
# 多头注意力机制将输入的x 在特征通道进行拆分
class Attention(nn.Module):
    def __init__(self,
                 inc, #输入特征
                 outc,
                 dropout = 0.5):
        # 输入 [B,N,C]:batchsize 序列长度 特征向量
        super(Attention, self).__init__()

        self.dim = inc ** (-0.5)

        self.layer = nn.LayerNorm(inc)
        self.to_qkv = nn.Sequential(nn.Linear(inc,inc*3),
                                    nn.Dropout())
        self.attn = nn.Dropout(dropout)
        self.linear = nn.Linear(inc,outc)
        #激活函数
        self.g = nn.GELU()

    def forward(self,x):
        x1 = self.layer(x)

        q,k,v = self.to_qkv(x1).chunk(chunks = 3,dim = -1)

        out = F.softmax((q@k.permute((0,2,1)))*self.dim,dim = -1)

        out = out @ v

        out = self.attn(out)
        # 线性层
        out = self.linear(out)
        x = self.g( out + x )

        return x


class MultiAttention(nn.Module):
    def __init__(self,
                 dim,
                 heads,
                 dropout = 0.5):
        super(MultiAttention, self).__init__()


        self.heads = heads
        self.hidden_dim = (heads * dim)

        self.attention = Attention(inc = dim, outc = dim,dropout = dropout)

        self.project = nn.Sequential(
                       nn.LayerNorm(dim),
                       nn.Linear(dim,self.hidden_dim),
                       nn.GELU(),
                       nn.Dropout(dropout),
                       nn.Linear(self.hidden_dim,self.hidden_dim)
        )

        self.out_proj = nn.Sequential(
                       nn.LayerNorm(self.hidden_dim),
                       nn.Linear(self.hidden_dim, dim),
                       nn.GELU(),
                       nn.Dropout(dropout)
        )


    def forward(self,x):

        features = []
        out = self.project(x)
        f = torch.chunk(out,chunks = self.heads,dim = -1)

        for feature in f:
         out = self.attention(feature)
         features.append(out)

        features = torch.cat(features,dim = -1)

        return self.out_proj(features)


# Encoder
class Transformer(nn.Module):
    def __init__(self,
                 dim, #输入通道
                 heads, #多头注意力
                 depth,#Transformer的深度
                 mlp_dim,#FFN
                 dropout = 0.5):
        super(Transformer, self).__init__()

        # 输出
        outc = 2 #

        # pos encoder
        self.pos = nn.Parameter(torch.randn((1,128*128,8)))

        self.layer = nn.ModuleList([])

        self.norm = nn.LayerNorm(dim)

        # to_out
        self.laent = nn.Sequential(
                   nn.LayerNorm(dim),
                   nn.Linear(dim,outc),
                   nn.GELU(),
                   nn.Dropout(dropout),
                   nn.Linear(outc,outc)

        )

        self.out = nn.Conv2d(in_channels = outc,
                             out_channels = outc,
                             kernel_size = 1,
                             stride = 1)

        for i in range(depth):
            self.layer.append(
                nn.ModuleList([
                    MultiAttention(dim = dim,heads = heads,dropout = dropout),
                    FFN(dim = dim,hidden_dim = mlp_dim,dropout = dropout)
                ])
            )

    def forward(self,x):


        x += self.pos #add position
        for attn,ffn in self.layer:
            # 注意力机制
            x = attn(x) + x
            x = self.norm(x)

            #前反馈网络
            x = ffn(x) + x
            x = self.norm(x)

        x = self.laent(x).permute((0,2,1))[:,:,:,None]
        return self.out(x)





if __name__ == '__main__':


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # x = torch.randn((1,15,1,1))
    #
    # model = illumination(inc = 15,
    #                      outc = 7)

    x = torch.randn((1, 4, 600, 400))
    # model = DeNoise_net(inc = 3,
    #                     outc = 3)

    # model = Fine_Noise(inc = 4,outc = 6)
    #
    # out = model(x)
    #
    # # 查看模型参数
    # flops_parameters(model = model,input = (x,))
    #
    # print(out.shape)

