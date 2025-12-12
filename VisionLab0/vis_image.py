# -------------------------------------#
# 更换模型
# -------------------------------------#
from nets.BuildRT import BuildRT
from nets.Bisenet import bisenetv1, bisenetv2
from nets.Pidnet import pidnet
from nets.STDC import model_stage
from nets.lpsnet import get_lspnet_m
from nets.FFNet.ffnet_gpu_small import segmentation_ffnet50_dAAA
from nets.DDRNet.DDRNet_23_slim import get_seg_model
from nets.unet import SegHead

# -----------------------------------------#
#           模块消融实验
# -----------------------------------------#
from nets.Albation_module import Albation

# Dice loss
from nets.criterion import Dice_loss,focal_loss,\
                           Heatmaploss
from datasets.dataloader import convert
from datasets.boundary import convert_boundary

import cv2
import os
import torch
import argparse
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F



class Grad_CAM(object):
    def __init__(self, modules = None,
                 pth_path = None,
                 num_classes = 2,
                 args = None):

        assert pth_path != None, '请输入权重...'
        assert args !=None,'请输入配置....'

        self.args = args # get varible

        method = args.method

        if (method == 'Ours'):
            self.net = BuildRT(n_classes=num_classes)

        elif method == 'bisenetv1':
            self.net = bisenetv1.BiSeNetV1(num_classes)
        elif method == 'ablation':
            self.net = Albation(num_classes, args=args)

        elif method == 'bisenetv2':
            self.net = bisenetv2.BiSeNetV2(num_classes)

        elif method == 'pidnet':
            self.net = pidnet.get_pred_model(name='pidnet_s', num_classes=num_classes)

        elif method == 'ddrnet':
            self.net = get_seg_model(pretrained=False, num_classes=num_classes, augment=False)

        elif method == 'ffnet':
            self.net = segmentation_ffnet50_dAAA()

        elif method == 'stdc':
            self.net = model_stage.BiSeNet('STDCNet813', num_classes, use_boundary_8=True)

        elif method == 'lpsnet':
            self.net = get_lspnet_m()

        elif method == 'unet':
            self.net = SegHead(num_classes)

        else:
            raise NotImplementedError('No exist Method!!!')


       # print(self.net)
      
        self.num_classes = num_classes

        self.input_shape = (args.h,args.w) #direct resize

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model = torch.load(pth_path, map_location = self.device)

        self.net.load_state_dict(model)

        self.net = self.net.eval()

        getattr(self.net, modules).register_forward_hook(self.__register_forward_hook)
        getattr(self.net, modules).register_backward_hook(self.__register_backward_hook)

        self.modules = modules

        # 保存梯度信息
        self.input_grad = [ ]
        # 收集feature map
        self.output_grad = [ ]

        # 特征
        self.feature_grad = []

    def __register_backward_hook(self,
                                 module,
                                 grad_in,

                                 grad_out):

        #print(len(grad_in), len(grad_out))

        self.input_grad.append(grad_out[ 0 ].detach().data.cpu().numpy())

    def __register_forward_hook(self,
                                module,
                                grad_in,
                                grad_out):
        self.output_grad.append(grad_out)
        self.feature_grad.append(grad_out.detach().data.cpu().numpy())

    def _get_cam(self, feature_map, grads):
        # -------------------------------------------------------#
        #                  feature_map: [c,h,w]
        #                  grads: [c,h,w]
        #                  return [h,w]
        # -------------------------------------------------------#
        cam = np.zeros(feature_map.shape[ 2: ], dtype = np.float32)
        alpha = np.mean(grads, axis = (2,3))


        for ind, c in enumerate(alpha):


            cam += c[ind] * feature_map[0][ ind ].detach().numpy()

        heatmap = np.maximum(cam, 0)


        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) + 1e-8)

        heatmap = cv2.resize(heatmap, self.input_shape)

        return heatmap

    def __get_cam(self,feature_map,grads):
        # -------------------------------------------------------#
        #                  feature_map: [c,h,w]
        #                  grads: [c,h,w]
        #                  return [h,w]
        # -------------------------------------------------------#
       # cam = np.zeros(feature_map.shape[2:], dtype=np.float32)
        print(grads.shape)
        cam = np.mean(grads, axis=(0,1))
        print(cam.shape)

        heatmap = np.maximum(cam, 0)


        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) + 1e-8)

        heatmap = cv2.resize(heatmap, self.input_shape)

        return heatmap

    def show_cam_to_image(self, image,
                          heatmap,
                          is_show = False,
                          is_write = False,
                          name = None):
        #heatmap = np.transpose(heatmap,(1,2,0))

        heatmap = np.array(heatmap * 255 , np.uint8)

       
        heatmap = cv2.applyColorMap(heatmap,
                           cv2.COLORMAP_JET)

        heatmap = np.float32(heatmap) / 255.


        image = np.transpose(image,(1,2,0))

        img =  0.4 * heatmap +  0.6 *np.array(image)

        # --------------------------------------------#
        #               clip pix value
        # --------------------------------------------#

        img = (img  - np.min(img)) / np.max(img)

        img = np.uint8(img * 255)

        if is_show:
            plt.imshow(img[ :, :, ::-1 ])
            plt.show()
        if is_write:
            cv2.imwrite(f'{args.save_path}\\{args.method}\\{name}_cam.jpg', img,[cv2.IMWRITE_JPEG_QUALITY,100])

    def vis_pix(self,pred, label,name):
        # ---------------------------------#
        # 预测 真实
        # 真实
        # TP: 预测为真，实际为真
        # FP：预测为真，实际为假
        # TN：预测为假，实际是假
        # FN：预测为假，实际是正
        # ---------------------------------#

        h, w,*_ = pred.shape

        # print(np.unique(np.reshape(pred, (-1,))))
        # print(np.unique(np.reshape(label, (-1,))))

        mask = np.ones((h, w, 3), dtype=np.uint8)

        color = np.array([147, 208, 80,  # 前景 绿
                          110, 147, 208,  # 蓝色
                          250, 250, 250,  # 背景 白
                          255, 230, 153])  # 黄色

        color = color.reshape((-1, 3))

        color = color[:, ::-1]

        mask[(label == 1) & (pred == 1)] = color[0]

        mask[(label == 0) & (pred == 1)] = color[1]

        mask[(label == 0) & (pred == 0)] = color[2]

        mask[(label == 1) & (pred == 0)] = color[3]

       # mask = Image.fromarray(mask)

        cv2.imwrite(f'{args.save_path}\\{args.method}\\{name}_pix_vis.png',mask)

    def forward(self, image,
                label,
                is_show = False,
                is_write = False,
                name = None):

        image = image.resize(self.input_shape)

        image.save(f'{args.save_path}\{args.method}\{name}.jpg')

       
        image = np.array(image, dtype = np.float32) / 255.
        image = np.transpose(image, (2, 0, 1))


        # 网络模型输入
        x = torch.from_numpy(image).float()
        x = x.unsqueeze(dim = 0)


        self.net.zero_grad() # 清空梯度

        self.net = self.net.eval()

        method = self.args.method
        h,w = args.h,args.w

        up = nn.Upsample(size=(h, w),
                         mode='bilinear', align_corners=True)

        if method == 'Ours':
            (feat_out8, feat_out16, feat_out32), \
                (feat_out_sp2, feat_out_sp4, feat_out_sp8, feat_out_sp16), \
                (pred_loc2, pred_loc4) = self.net(x)
           # (feat_out8) = self.net(x)



        elif (method == 'bisenetv1') | (method == 'bisenetv2') | (method == 'unet') | (method == 'ablation'):
            out = self.net(x)
            feat_out8 = out[0]
        elif (args.method == 'ddrnet') | (args.method == 'ffnet') | (args.method == 'lpsnet') \
                | (args.method == 'pidnet'):

            feat_out8 = self.net(x)

            if (args.method == 'pidnet'):
                feat_out8 = feat_out8[0]

            # output = up(output[0])
            # TODO ?

            feat_out8 =up(feat_out8)

        elif args.method == 'stdc':

            *output, boundary = self.net(x)

            feat_out8 = up(output[0])

            # ---------------------------------#
        #            损失函数定义
        # ---------------------------------#

       # b,c,h,w = feat_out8.shape
        #label = torch.ones((b,c,h,w),requires_grad = True).float()
        #print(output_main.shape,label.shape)



        #--------------------------------------------#
        #                   取值
        #--------------------------------------------#
       


        
        feat_out8 = F.softmax(feat_out8,dim = 1)


        unmask_main = torch.argmax(feat_out8,dim = 1)

        unmask = ( unmask_main)


        #---------------------------------------------------------#
        #  真实标签
        #---------------------------------------------------------#
        _label = np.array(label)
        #print(_label.shape)

        _label[_label == 0] = 0
        _label[_label != 0] = 1


        #----------------------------------------------------------#
        #   生成的标签
        #-----------------------------------------------------------#
        glabel = unmask.detach().cpu().numpy()[0] #取出

        T_mask = np.zeros((self.input_shape[1], self.input_shape[0], self.num_classes))

        # --------------------------------------#
        # 两种构建one-hot编码形式
        # --------------------------------------#
        for c in range(self.num_classes):
            T_mask[glabel == c, c] = 1
        T_mask = np.transpose(T_mask, (2, 0, 1))
        T_mask = convert(T_mask) #dice label
        T_mask = torch.unsqueeze(T_mask,dim=0)

        heat_map2,boundary2 = convert_boundary(label,2)
        heat_map4,boundary4 = convert_boundary(label,4)
        _,boundary8 = convert_boundary(label,8)

        heat_map2 = convert(np.transpose(heat_map2, (2, 0, 1)))
        heat_map4 = convert(np.transpose(heat_map4, (2, 0, 1)))
        heat_map2 = torch.unsqueeze(heat_map2, dim=0)
        heat_map4 = torch.unsqueeze(heat_map4, dim=0)

        boundary2 = convert(np.transpose(boundary2, (2, 0, 1)))
        boundary4 = convert(np.transpose(boundary4, (2, 0, 1)))
        boundary8 = convert(np.transpose(boundary8, (2, 0, 1)))

        boundary2 = torch.unsqueeze(boundary2, dim=0)

        boundary4 = torch.unsqueeze(boundary4, dim=0)
        boundary8 = torch.unsqueeze(boundary8, dim=0)

        glabel = torch.from_numpy(glabel).unsqueeze(dim=0).long()






        heatloss = Heatmaploss() #没有改
        criteror = nn.BCEWithLogitsLoss()
        with torch.no_grad():
            # 二值图
            self.vis(unmask.detach().cpu().numpy(),name)
            # label = np.array(label)
            #self.vis(_label, name)

            # 色差图
            #self.vis_pix(unmask.detach().cpu().numpy()[0],_label,name)

            #self.vis_entropy(feat_out8.detach().cpu(),name)



       #self.net = self.net.train()
        #----------------------------------------------------------#
        #                 损失函数和训练时保持一致
        #          TODO?
        #----------------------------------------------------------#
        # 边界自蒸馏损失
       #  _feat = up(feat_out_sp4)
       #  _kl_s = nn.MSELoss()(F.sigmoid(up(feat_out_sp2)), F.sigmoid(_feat))
       #
       #
       #  # 分割自蒸馏损失
       #  _seg_kl1 = nn.KLDivLoss()(F.log_softmax(feat_out16, dim=1), F.softmax(feat_out8, dim=1))
       #  _seg_kl2 = nn.KLDivLoss()(F.log_softmax(feat_out8, dim=1), F.softmax(feat_out16, dim=1))
       #  _seg_kl = (_seg_kl1 + _seg_kl2) * args.A
       #
       #
       #  #-------------------------------------------------------------------------------------#
       #  #  分割损失
       #  #-------------------------------------------------------------------------------------#
       #  #with torch.no_grad():
       #  alpha = (feat_out8 - T_mask).mean()
       #  alpha = alpha ** 2
       #
       #  segloss8 = Dice_loss(feat_out8, T_mask) + nn.CrossEntropyLoss()(feat_out8, glabel)
       #  segloss16 = Dice_loss(feat_out16, T_mask) + nn.CrossEntropyLoss()(feat_out16, glabel)
       #  segloss32 = Dice_loss(feat_out32, T_mask) + nn.CrossEntropyLoss()(feat_out32, glabel)
       #
       #  segloss = (alpha * segloss8 + (segloss16 + segloss32) + _seg_kl)
       #
       #
       #  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
       #  # 边界损失
       #  # 标签
       # # with torch.no_grad():
       #  bate = (feat_out_sp2 - boundary2).mean() ** 2
       #  boundaryloss2 = heatloss(feat_out_sp2, boundary2)
       #  boundaryloss4 = heatloss(feat_out_sp4, boundary4)
       #  boundaryloss8 = criteror(feat_out_sp8, boundary8)
       #
       #  boundaryloss = (bate * boundaryloss2 + boundaryloss4 + boundaryloss8)
       #
       #
       #  #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
       #  # 定位损失
       #  loc2loss = focal_loss(pred_loc2, heat_map2, args.alpha, args.bate)
       #  loc4loss = focal_loss(pred_loc4, heat_map4, args.alpha, args.bate)
       #  locloss = (loc2loss + args.L * loc4loss)
       #
       #  Tloss = (boundaryloss + segloss + _kl_s + locloss)
       #  #Tloss.backward()


        #generate CAM
        # grad = self.input_grad[0]
        #
        # fmap = self.output_grad[0]

        fmap = self.feature_grad[0]

        #cam = self._get_cam(fmap,grad)
        cam = self.__get_cam(fmap,fmap)

        # show
        image = np.float32(image)


        # self.show_cam_to_image(image,cam,is_show ,
        #                        is_write,name)

        self.input_grad.clear()
        self.output_grad.clear()

    def vis(self,mask,name):
    #------------------------------#
    #           可视化
    #------------------------------#
     
     mask = mask[0]
    
     platte = [0, 0, 0, 255, 255, 255, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153,
                              250,
                              170, 30,
                              220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142,
                              0,
                              0, 70,
                              0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

     for i in range(256 *3 - len(platte)):
         platte.append(0)
    
     platte = np.array(platte,dtype = np.uint8)

     mask = np.array(mask,dtype = np.uint8)

     mask = Image.fromarray(mask).convert('P')

     

     mask.putpalette(platte)

     

     #mask.show()

     mask.save(f'{args.save_path}\\{args.method}\\{name}_vis.png')

    def prob2entropy(self,prob):


        b,c,h,w = prob.shape

        x = -torch.mul(prob,torch.log2(prob + 1e-20)) / np.log2(c)

        return x
  
    def vis_entropy(self,
                    prob,
                    name):
    #----------------------------------------#
    #               熵值结果
    #----------------------------------------#
        entropy = self.prob2entropy(F.softmax(prob,dim=1)) #
       # entropy = F.softmax(prob,dim=1)

        entropy = entropy[0].detach().cpu().numpy()
      

        

        entropy_background = np.array(entropy[0, ...] * 255)
        entropy_foreground = np.array(entropy[1, ...] * 255)
        #
        heatmap = (0.5 * entropy_background + 0.5 * entropy_foreground)

       # heatmap = (heatmap - np.max(heatmap)) / (np.min(heatmap))

        heatmap = np.array(heatmap , np.uint8)
        entropy = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        

    
        #cv2.imshow('entropy',entropy)
      
        cv2.imwrite(f'{args.save_path}\{args.method}\{name}_entropy.png',entropy)
        #cv2.waitKey(0)






if __name__ == '__main__':


   args = argparse.ArgumentParser(description = 'inference....')

   args.add_argument('--image_path','--i',default = '',help = 'inference image path...')
   args.add_argument('--model_path','--m',default = '',help = 'inference model path...')


   #-----------------------------------------------------------------------------------------#
   #            增加文件读取
   #-----------------------------------------------------------------------------------------#
   args.add_argument('--val_txt','--vt',required = True,help='inference txt....')
   args.add_argument('--vis_layer','--v',default = 'conv_out',help = 'vis layer...')
   args.add_argument('--is_show',action = 'store_true',help = 'Using vis image...')
   args.add_argument('--is_write',action = 'store_true',help = 'Using save vis...')
   args.add_argument('--save_path',default='vis',type = str,help = 'save inferece result path(source domain ->  target domain)...',required = True)
   args.add_argument('--method',type = str,required = True)

   #------------------------------------------------------------------------------------------#
   #                                文件夹推理
   #------------------------------------------------------------------------------------------#
   args.add_argument('--dir',type = str,default = '')


   #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
   #    h,w
   #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
   args.add_argument('--h',type=int,default=512,help='inference image h...')
   args.add_argument('--w', type=int, default=512, help='inference image w...')


   #
   # 模块消融实验
   args.add_argument('--flow', '--f', action='store_true',
                      help='Multi-scale feature aggregation')
   args.add_argument('--output', '--out', action='store_true',
                      help='Feature modification')
   args.add_argument('--block', '--bl', action='store_true',
                      help='SDCM Block...')

   # 超参数
   args.add_argument('--A', type=float, default=0.1,
                              help='Self-distillation segmentation alignment loss coefficient')

   args.add_argument('--L', type=float, default=0.01,
                              help='Location loss factor')

   args.add_argument('--alpha', type=float, default=0.05,
                              help='classifier...')
   args.add_argument('--bate', type=float, default=0.5,
                              help='classifier...')

   args = args.parse_args()

   path = args.image_path

   model_path = args.model_path
   

   #----------------------------------------------------------#
   #                    
   #----------------------------------------------------------#
   # from glob import glob
   #
   # images = glob(f'{args.dir}\*.tiff')


   images = list(map(lambda x:x.strip().split(' ')[0],open(args.val_txt).readlines()))



   if not os.path.exists(args.save_path):
      os.makedirs(args.save_path)


   if not os.path.exists(f'{args.save_path}\\{args.method}'):
      os.makedirs(f'{args.save_path}\\{args.method}')

   for path in images:

    image = Image.open(path).convert('RGB')

    label_path = path.replace('image','label').replace('tiff','tif')
    label = Image.open(label_path).convert('L')

    # 每个模型需要获取的moduel name都不一样
    cam = Grad_CAM(modules = args.vis_layer,pth_path=model_path,args=args)

    

    cam.forward(image=image,label=label,is_show = args.is_show,is_write = args.is_write,name = os.path.basename(path).split('.')[0])