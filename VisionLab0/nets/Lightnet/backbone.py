import torch
import torch.nn as nn
from  torchsummary import torchsummary
from nets.Lightnet.models import CBA,\
    LA_Block,Stemblock,HP_Stage


#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class Conv(nn.Module):
    def __init__(self, c1, c2, k, s = 1, p = 0, d = 1, g = 1, leaky = True):
        super(Conv, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(c1, c2, k, stride = s, padding = p, dilation = d, groups = g)
        )

    def forward(self, x):
        return self.convs(x)



class Encoder_Model(nn.Module):
    def __init__(self,num_class,use_cuda = True):
        super(Encoder_Model, self).__init__()
        self.encoder_Stembackbone = Stemblock(num_class = 3)
        self.encoder_Hpbackbone = HP_Stage()
        #self.__init__weight()
        self.use_cuda(use_cuda )
    def use_cuda(self, use_cuda = False):
        self.encoder_Hpbackbone.cuda()
        self.encoder_Stembackbone.cuda()
    def forward(self,x):
        low_level,feature = self.encoder_Stembackbone(x)
        low_feature = [low_level,feature]
        output = self.encoder_Hpbackbone(feature)
        out = low_feature + output
        return out
    def __init__weight(self):
        for layer in self.encoder_Stembackbone.modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, mode = 'fan_out',
                                              nonlinearity = 'relu')
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, val = 0.0)
            elif isinstance(layer, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(layer.weight, val = 1.0)
                torch.nn.init.constant_(layer.bias, val = 0.0)
        for layer in self.encoder_Hpbackbone.modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, mode = 'fan_out',
                                              nonlinearity = 'relu')
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, val = 0.0)
            elif isinstance(layer, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(layer.weight, val = 1.0)
                torch.nn.init.constant_(layer.bias, val = 0.0)





#上采样层
class Cascade_block(nn.Module):
    def __init__(self,s =2,cfg = 0):
        super(Cascade_block,self).__init__()
        self.Up = nn.Sequential(
            nn.Upsample(scale_factor = s,mode = 'bilinear', align_corners = True),
            nn.ReLU(inplace = True)
        )
        total_channel = [1024 + 512,256+512,256+128,128+64]
        out_channel = [512,256,128,64]
        self.conv = CBA(c = total_channel[cfg],outc = out_channel[cfg],
                        use_lab = False)
    def forward(self,x,x1):
        #x是低维信息 x1是高维信息
        out = x
        x1 = self.Up(x1)
        x = torch.cat((x,x1),dim = 1)
        x = self.conv(x)
        return torch.add(out,x)
#解码器
class Decoder(nn.Module):
    def __init__(self,numsclass = 3):
        super(Decoder, self).__init__()
        #不进行参数共享
        self.layer1 = Cascade_block(cfg =  0 )
        self.layer2 = Cascade_block(cfg = 1)
        self.layer3 = Cascade_block(cfg = 2)
        self.layer4 = Cascade_block(cfg = 3)
        self.output = nn.Sequential(
            CBA(c = 64,outc = 32,k = 1,use_lab = False),
            CBA(c = 32, outc = numsclass, k = 1,use_lab = False)
        )
        self.la1 = LA_Block(channel =256 ,x= 128,y=128)
        self.la2 = LA_Block(channel = 128, x = 256, y = 256)
        self.la3 = LA_Block(channel = 64, x = 512, y = 512)
        #self.__init__weight()

    def forward(self,output):
        x1,x2,x3,x4,x5 = output

        c1 = self.layer1(x4,x5)#128 128 58

        c2 = self.layer2(x3,c1)#32 32 232
        c2 = self.la1(c2)

        c3 = self.layer3(x2,c2)

        c3 = self.la2(c3)

        out = self.layer4(x1,c3)#128 128 58
        out = self.la3(out)
        out = self.output(out)
        return out
    def __init__weight(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, mode = 'fan_out',
                                              nonlinearity = 'relu')
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, val = 0.0)
            elif isinstance(layer, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(layer.weight, val = 1.0)
                torch.nn.init.constant_(layer.bias, val = 0.0)


class LightSeg(nn.Module):
    def __init__(self,num_classes = 3):
        super(LightSeg, self).__init__()
        self.encoder = Encoder_Model(num_class=num_classes)
        self.decoder = Decoder(numsclass = num_classes)  # 分类类别
        self.use_cuda(use_cuda = True)
    def use_cuda(self, use_cuda = True):
        # 使用cuda
        #self.encoder(use_cuda = True)
        self.decoder.cuda()
    def forward(self,x):
        out = self.encoder(x)
        x = self.decoder(out)
        return x

if __name__=="__main__":
    x = torch.rand((1,3,512,512)).cuda()
    model = LightSeg(num_classes = 3)
  
    #y = model.forward(x)
    #torch.save(model.state_dict(),'1.pth')
    #print(y.shape)

    from nets.USI import flops_parameters

    flops_parameters(model,(x,))

    # decoder =Decoder().cuda()
    # #print("---------编码器---------------")
    #torchsummary.summary(model,(3,512,512),2,"CPU")
    # encoder = Encoder(numsclass = 5).cuda()#分类类别
    # x = decoder(x)
    #
    # output = encoder(x).cuda()
    # print(output.shape)