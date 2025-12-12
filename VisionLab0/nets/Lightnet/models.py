import torch.nn as nn
import  torch
import torch.nn.functional as F
from torch.nn import Parameter

#单独为这一层设置优化策略
class LearnableAffineBlock(nn.Module):
    def __init__(self,
                 scale_value=1.0,
                 bias_value=0.0,
                 lr_mult=1.0,
                 lab_lr=0.01):
        super().__init__()
        self.scale = Parameter(torch.tensor([scale_value]),requires_grad = True)
        self.bias = Parameter(torch.tensor([bias_value]),requires_grad = True)
        # self.scale = self.create_parameter(
        #     shape=[1, ],
        #     default_initializer=Constant(value=scale_value),
        #     attr=ParamAttr(learning_rate=lr_mult * lab_lr))
        # self.add_parameter("scale", self.scale)
        # self.bias = self.create_parameter(
        #     shape=[1, ],
        #     default_initializer=Constant(value=bias_value),
        #     attr=ParamAttr(learning_rate=lr_mult * lab_lr))
        # self.add_parameter("bias", self.bias)

    def forward(self, x):
        return self.scale * x + self.bias


class Stemblock(nn.Module):
    def __init__(self,num_class,M = 64):
        super(Stemblock,self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels = num_class,out_channels = M,kernel_size = 3,padding = 1),
        )

        self.layer1 =  nn.Sequential(
            nn.Conv2d(in_channels = num_class,out_channels = M,kernel_size = 1),
        )
        self.left = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2,stride = 2)
        )
        self.right = nn.Sequential(
            nn.Conv2d(in_channels = M,out_channels = M//2,kernel_size = 3,bias = False,padding = 1),
            nn.BatchNorm2d(M//2),
            nn.Conv2d(in_channels = M // 2, out_channels = M ,kernel_size = 3,stride = 2,padding = 1,bias = False),
            nn.BatchNorm2d(M)
        )
        self.act = nn.ReLU(inplace = True)
    def forward(self,x,use_act = True):
        output = []
        x = self.layer(x)
        output.append(x)
        left = self.left(x)
        right = self.right(x)
        x = torch.cat((left,right),dim = 1)
        #增加通道混洗，增强特征
        x = channel_shuffle(x,4)
        x = self.act(x)
        output.append(x)
        return output

stage = {128:256,
         256:512,
         512:1024}
#h，w和channel一样
class HP_block(nn.Module):
    def __init__(self,inc,h,w):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size = 2,stride = 2)
        self.layer1 = CBA(inc,stage[inc],k=2,s=2)
        self.layer2 = CBA(inc, stage[inc], k = 3, s = 2,p=1)
        self.layer3 = CBA(inc, stage[inc], k = 5, s = 2,p = 2)

        self.conv = nn.Sequential(
            nn.Conv2d(stage[inc]*3,stage[inc]*2,kernel_size = 1),
            nn.Conv2d(stage[ inc ] * 2, inc  , kernel_size = 1)
        )
        self.la = Attention(stage[ inc ])
    def forward(self,x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        left = self.pool(x)
        x = torch.cat([x1,x2,x3],dim = 1)
        x = self.conv(x)
        x = torch.cat([left,x],dim = 1)
        #这个注意力机制有点小问题
        x = self.la(x)
        x = channel_shuffle(x ,4)
        return x

class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, leaky=True):
        super(Conv, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g)
        )

    def forward(self, x):
        return self.convs(x)

def conv1x1(k=1,inc = None,outc = None):
    return nn.Conv2d(in_channels = inc,out_channels = outc,kernel_size = k)

#权重初始化
class CBA(nn.Module):
    def __init__(self,c=None,outc = 1,k=1,s=1,p=0,g = 1,use_lab = True):
        super(CBA,self).__init__()
        self.use_lab = use_lab
        self.conv = nn.Conv2d(in_channels = c ,out_channels = outc,kernel_size = k,stride = s,padding = p,groups = g,bias = False)
        self.bn = nn.BatchNorm2d(outc)
        self.act = nn.ReLU(inplace = True)
        if self.use_lab:
            self.lab = LearnableAffineBlock()
    def forward(self,x):
       x = self.conv(x)
       x = self.bn(x)
       x = self.act(x)
       if self.use_lab:
           x = self.lab(x)
       return  x

#shufflenetv2
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class Attention(nn.Module):
    def __init__(self,inc):
        super(Attention, self).__init__()
        self.layer1 = CBA(inc, inc, k = 3, s = 1,p=1)
        self.layer2 = CBA(inc, inc, k = 5, s = 1, p = 2)
        self.layer3 = CBA(inc, inc, k = 7, s = 1, p = 3)
        self.dk = nn.Parameter(torch.tensor([1.]),requires_grad = True)
    def forward(self,x):
        idinity = x
        Q = self.layer1(x)
        K = self.layer2(x)
        V = self.layer3(x)
        result = idinity + idinity * F.softmax(torch.div((Q@K),self.dk),dim = 1)*V
        return result



#全局上下文信息
class LA_Block(nn.Module):
    """
    输入原特征图的通道信息 reduction是通道拆分倍数
    x,y是输入特征图的尺度信息
    """
    def __init__(self, channel,x,y, reduction=8):
        super(LA_Block, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

        self.conv = CBA(c = channel//reduction,outc = channel,k = 1,use_lab = False)


        self.avg_pool_x = nn.AdaptiveAvgPool2d((x , 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, y))
        self.conv1X1 = CBA(c= channel, outc = channel // reduction, k = 1,use_lab = False)

        self.sigmoid_x = nn.Sigmoid()
        self.conv2 = Conv(c1=channel, c2=channel, k=1, s=1)

    def forward(self, x):
        idenity = x
        x, y = self.avg_pool_x(x), self.avg_pool_y(x)
        x, y = self.conv1X1(x), self.conv1X1(y)
        out = x@y
        x_sig = self.sigmoid_x(out)
        x_conv = self.conv2(self.conv(x_sig))
        return x_conv + idenity

class HP_Stage(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage = []
        size = 128
        for i,v in stage.items():
            self.stage.append(HP_block(i,size,size))
            size*=2
        self.feature = nn.Sequential(*self.stage)
    def forward(self,x):
        output = []
        for feature in self.feature:
            x = feature(x)
            output.append(x)
        return output


class PPM(nn.Module):
    def __init__(self,inc):
        super(PPM,self).__init__()
        self.cfg = [1,3,5,7]
        self.conv = conv1x1(k=1,inc = inc
                            ,outc = inc//4)
        self.convout = conv1x1(k=1,inc = 2*inc,outc = inc)
        self.attention = Attention(inc)
    def pool(self,x,size):
        adaptivepool = F.adaptive_avg_pool2d(x,size)
        return adaptivepool
    def upsample(self, x, size):  # 上采样使用双线性插值
        return F.interpolate(x, size, mode = 'bilinear', align_corners = True)

    def forward(self,x):
        identity = x
        size = x.shape[2:]
        feat1 = self.pool(x,self.cfg[0])
        feat1 = self.conv(feat1)
        feat2 = self.pool(x, self.cfg[ 1 ])
        feat2 = self.conv(feat2)
        feat3 = self.pool(x, self.cfg[ 2 ])
        feat3 = self.conv(feat3)
        feat4 = self.pool(x, self.cfg[ 3 ])
        feat4 = self.conv(feat4)

        #恢复到原特征图尺度 前三个作为Q，K，V
        feat1 = self.upsample(feat1,size)
        feat2 = self.upsample(feat2,size)
        feat3 = self.upsample(feat3,size)

        feat4 = self.upsample(feat4,size)

        out = self.conv(x)

        feat1 = torch.mul(feat1,out) + out
        feat2 = torch.mul(feat2, out) + out
        feat3 = torch.mul(feat3,out) + out
        feat4 = torch.mul(feat4, out) + out

        x = torch.cat((feat1,feat2,feat3,feat4,x),dim = 1)

        x = self.attention(x)
        x = self.convout(x) + identity
        return x


if __name__=="__main__":
    x = torch.rand((1,128,256,256))
    model = LA_Block(channel = 128,x = 256,y =256)
    y = model(x)
    print(y.shape)
