# from tqdm import tqdm
# import torch
# import numpy as np
# from nets.criterion import Dice_loss,score,\
#      Heatmaploss,focal_loss
# import torch.nn as nn
# from utils.tools import get_lr
# from utils.tools import prob2entropy
# import torch.nn.functional as F
# from utils.config import Config
#
#
# def epoch_fit(cur_epoch,total_epoch,save_step,
#               model,optimizer,dataloader,
#               device,logger,args,save_path,
#               eval,local_rank): #评估函数是对测试集
#
#
#
#     model = model.to(device)
#
#     if torch.cuda.device_count() > 1:
#         #-----------------------------------------------------#
#         #                  训练的BS要能被显卡数整除
#         #-----------------------------------------------------#
#         #print(f'GPUS Numbers :{torch.cuda.device_count()}')
#
#         #-------------------------------------------------------#
#         #                         设置多卡训练
#         #-------------------------------------------------------#
#         model = nn.DataParallel(model)
#
#
#
#     traindataloader,testdataloader,valdataloader = dataloader
#
#
#     # 损失函数定义
#     criteror = nn.BCEWithLogitsLoss()
#     heatloss = Heatmaploss() # TODO ? loc loss ?
#     # 上采样
#     up = nn.Upsample(size = args.input_shape,
#                      mode = 'bilinear', align_corners = True)
#
#     with tqdm(desc = f"Epoch: [{cur_epoch + 1}]/[{total_epoch}]",
#               total = len(traindataloader),mininterval = 0.3,postfix = dict,colour = '#6DB9EF') as pb:
#
#         model = model.train()
#
#         #-------------------------------------#
#         # 训练损失计算
#         #-------------------------------------#
#         total_loss = 0.
#         total_cls_loss = 0.
#         total_boundary_loss = 0.
#         total_heat_loss = 0.
#
#         # 源域
#         for ind,batch in enumerate(traindataloader):
#           image,label,png,(heat2,heat4),((boundary1,boundary2),boundary4,boundary8,boundary16) = batch
#           with torch.no_grad():
#               # TODO 1
#               image = image.to(device)
#               label = label.to(device)
#               png = png.to(device)
#
#               # TODO 2 heatmap
#               heat2 = heat2.to(device)
#               heat4 = heat4.to(device)
#
#               # TODO 3 boundary
#               boundary2 = boundary2.to(device)
#               boundary4 = boundary4.to(device)
#               boundary8 = boundary8.to(device)
#               boundary16 = boundary16.to(device)
#
#
#           if args.method == 'Ours':
#               (feat_out8, feat_out16, feat_out32), \
#               (feat_out_sp2, feat_out_sp4, feat_out_sp8, feat_out_sp16), \
#               (pred_loc2, pred_loc4)  = model(image)
#
#               #------------------------------------------------------------------------------------#
#               # TODO 1 seg
#               #------------------------------------------------------------------------------------#
#               segloss8 = Dice_loss(feat_out8,label) + nn.CrossEntropyLoss()(feat_out8,png)
#               segloss16 = Dice_loss(feat_out16,label) + nn.CrossEntropyLoss()(feat_out16,png)
#               segloss32 = Dice_loss(feat_out32,label) + nn.CrossEntropyLoss()(feat_out32,png)
#
#               segloss = (segloss8 + 0.5 * segloss16 + 0.2* segloss32)
#               #--------------------------------------------------------------------------#
#               # TODO 2 boundary
#               #--------------------------------------------------------------------------#
#               boundaryloss2 = criteror(feat_out_sp2,boundary2)
#               boundaryloss4 = criteror(feat_out_sp4,boundary4)
#               boundaryloss8 = criteror(feat_out_sp8,boundary8)
#               boundaryloss16 = criteror(feat_out_sp16,boundary16)
#
#               boundaryloss = (boundaryloss2 + 0.5 * boundaryloss4 + 0.1 * boundaryloss8 + boundaryloss16)
#
#               # ------------------------------------------------------------------------------------#
#               # TODO 3 loc
#               # ------------------------------------------------------------------------------------#
#               loc2loss = focal_loss(pred_loc2,heat2)
#               loc4loss = focal_loss(pred_loc4,heat4)
#               locloss = (loc2loss + 0.2 * loc4loss)
#
#               Tloss = (segloss + boundaryloss + locloss)
#
#               # -------------------------------------------------------------------------#
#               # TODO
#               # -------------------------------------------------------------------------#
#               total_cls_loss += (segloss).item()
#               total_boundary_loss += (boundaryloss).item()
#               total_heat_loss += (locloss).item()
#
#               total_loss += (total_heat_loss + total_cls_loss + \
#                              total_boundary_loss)
#
#           elif (args.method == 'bisenetv1' )|(args.method == 'bisenetv2')\
#                   |(args.method == 'unet'):
#
#                output = model(image)
#
#
#
#                ds = [] #dice loss
#                cs = [] #ce loss
#
#                for o in output:
#
#                    # 上采样
#                    #o = up(o)
#
#                    ds.append(Dice_loss(o,label))
#                    cs.append(nn.CrossEntropyLoss()(o,png))
#
#                total_cls_loss += (sum(ds).item() + sum(cs).item())
#                total_loss += total_cls_loss
#
#                Tloss = sum(ds) + sum(cs)
#
#           elif (args.method == 'ddrnet' )| (args.method == 'ffnet') | (args.method == 'lpsnet'):
#
#               output = model(image)
#
#               output = up(output)
#
#               loss1 = Dice_loss(output, label)
#               loss2 = nn.CrossEntropyLoss()(output, png)
#
#               total_cls_loss += (loss1 + loss2).item()
#
#               total_loss += total_cls_loss
#
#               Tloss = (loss1 + loss2)
#
#           elif (args.method == 'stdc') | (args.method == 'pidnet'):
#               ds = [ ]  # dice loss
#               cs = [ ]  # ce loss
#
#               *output,boundary = model(image)
#
#               for o in output:
#                   if args.method == 'pidnet':
#                      o = up(o)
#                   ds.append(Dice_loss(o, label))
#                   cs.append(nn.CrossEntropyLoss()(o, png))
#
#               if args.method == 'stdc':
#                   _boundaryloss = criteror(boundary,boundary8)
#               else:
#                   boundary = up(boundary)
#                   _boundaryloss = criteror(boundary, boundary1)
#               #stdc_bundaryloss = criteror(boundary,boundary8)
#               total_boundary_loss += _boundaryloss.item()
#               total_cls_loss += (sum(ds).item() + sum(cs).item())
#               total_loss += (total_cls_loss + total_boundary_loss)
#
#               Tloss = sum(ds) + sum(cs) + _boundaryloss
#
#           Tloss = Tloss / args.accumulate
#
#
#           Tloss.backward()
#           if ((ind + 1 ) % args.accumulate) == 0:
#
#               optimizer.step()
#
#               optimizer.zero_grad()
#
#
#           pb.set_postfix(**{
#               'total_loss': total_loss/(ind + 1),
#               'total_cls_loss':total_cls_loss/(ind + 1),
#               'total_boundary_loss':total_boundary_loss/(ind + 1),
#               'total_heat_loss':total_heat_loss/(ind + 1)
#           })
#
#           logger.info(f'total_loss: {total_loss/(ind + 1)} total_cls_loss: {total_cls_loss/(ind + 1)}')
#           logger.info(f'total_boundary_loss: {total_boundary_loss/(ind + 1)}')
#           logger.info(f'total_heat_loss: {total_heat_loss/(ind + 1)}')
#
#
#           pb.update(1)
#
#
#     #----------------------------------------------------------------------------#
#     # TODO freeze BN
#     #----------------------------------------------------------------------------#
#     model = model.eval()
#
#     if (cur_epoch % save_step) == 0:
#         #Evaluatation
#         with tqdm(desc = f"Epoch: [{cur_epoch + 1}]/[{total_epoch}]",
#                   total = len(valdataloader),mininterval = 0.3,postfix = dict,colour = '#7E89EF') as pb:
#
#          with torch.no_grad():
#             # TODO
#             for ind,batch in enumerate(valdataloader):
#                 image,label,png,(_,_),*_ = batch
#
#                 image = image.to(device)
#
#                 png = png.to(device).detach().cpu().numpy()
#
#                 if args.method == 'Ours':
#                     (feat_out8, feat_out16, feat_out32), \
#                     (feat_out_sp2, feat_out_sp4, feat_out_sp8, feat_out_sp16), \
#                     (pred_loc2, pred_loc4) = model(image)
#
#                     OneHotlabel8 = F.softmax(feat_out8, dim=1)
#                     #OneHotlabel16 = F.softmax(feat_out16, dim = 1)
#                     #OneHotlabel32 = F.softmax(feat_out32, dim = 1)
#
#                     targetOneHotlabel8 = torch.argmax((OneHotlabel8), dim=1).detach().cpu().numpy()  # 目标域
#                     #targetOneHotlabel16 = torch.argmax((OneHotlabel16), dim = 1).detach().cpu().numpy()  # 目标域
#                     #targetOneHotlabel32 = torch.argmax((OneHotlabel32), dim = 1).detach().cpu().numpy()  # 目标域
#                     targetlabel = targetOneHotlabel8
#                     #targetlabel = (targetOneHotlabel8 | targetOneHotlabel16 | targetOneHotlabel32)
#
#                 elif (args.method == 'ddrnet') | (args.method == 'ffnet') |(args.method == 'lpsnet'):
#                     output = model(image)
#
#                     pre = F.softmax(up(output), dim = 1)
#
#                     targetOneHotlabel8 = torch.argmax(pre, dim = 1).detach().cpu().numpy()
#
#                     targetlabel = targetOneHotlabel8
#                 elif (args.method == 'bisenetv1' )|(args.method == 'bisenetv2' )\
#                         |(args.method == 'unet'):
#
#                     output = model(image)
#
#                     pre = F.softmax((output[1]),dim = 1)
#
#                     targetOneHotlabel8 = torch.argmax(pre, dim=1).detach().cpu().numpy()
#
#                     targetlabel = targetOneHotlabel8
#
#                 elif (args.method == 'stdc') | (args.method == 'pidnet'):
#
#                     *out,_ = model(image)
#                     pre = F.softmax(up(out[ 0 ]) if (args.method == 'pidnet') else out[0], dim = 1)
#
#                     targetOneHotlabel8 = torch.argmax(pre, dim = 1).detach().cpu().numpy()
#
#                     targetlabel = targetOneHotlabel8
#
#                 eval.init(png,targetlabel)
#
#                 pb.update(1)
#
#             # 评估结果
#             eval.show()
#
#             res = eval.get_res()
#
#             logger.info(f'Epoch: {cur_epoch}  res: {res}')
#
#     if ((cur_epoch + 1)%save_step == 0) :
#             torch.save(model.module.state_dict(),f'{save_path}/{(cur_epoch + 1)}.pth')
#     if (cur_epoch + 1) == total_epoch:
#      torch.save(model.module.state_dict(),f'{save_path}/last.pth')


# # from tqdm import tqdm
# # import torch
# # import numpy as np
# # from nets.criterion import Dice_loss,score,\
# #      Heatmaploss,focal_loss
# # import torch.nn as nn
# # from utils.tools import get_lr
# # from utils.tools import prob2entropy
# # import torch.nn.functional as F
# # from utils.config import Config


# # def epoch_fit(cur_epoch,total_epoch,save_step,
# #               model,optimizer,dataloader,
# #               device,logger,args,save_path,
# #               eval,local_rank): #评估函数是对测试集


# #     model = model.to(device)

# #     if torch.cuda.device_count() > 1:
# #         #-----------------------------------------------------#
# #         #                  训练的BS要能被显卡数整除
# #         #-----------------------------------------------------#
# #         #print(f'GPUS Numbers :{torch.cuda.device_count()}')

# #         #-------------------------------------------------------#
# #         #                         设置多卡训练
# #         #-------------------------------------------------------#
# #         model = nn.DataParallel(model)


# #     traindataloader,testdataloader,valdataloader = dataloader


# #     # 损失函数定义
# #     criteror = nn.BCEWithLogitsLoss()
# #     heatloss = Heatmaploss() # TODO ? loc loss ?
# #     # 上采样
# #     up = nn.Upsample(size = args.input_shape,
# #                      mode = 'bilinear', align_corners = True)

# #     with tqdm(desc = f"Epoch: [{cur_epoch + 1}]/[{total_epoch}]",
# #               total = len(traindataloader),mininterval = 0.3,postfix = dict,colour = '#6DB9EF') as pb:

# #         model = model.train()

# #         #-------------------------------------#
# #         # 训练损失计算
# #         #-------------------------------------#
# #         total_loss = 0.
# #         total_cls_loss = 0.
# #         total_boundary_loss = 0.
# #         total_heat_loss = 0.
# #         total_kl_loss = 0.

# #         # 源域
# #         for ind,batch in enumerate(traindataloader):
# #           image,label,png,(heat2,heat4),((boundary1,boundary2),boundary4,boundary8,boundary16) = batch
# #           with torch.no_grad():
# #               # TODO 1
# #               image = image.to(device)
# #               label = label.to(device)
# #               png = png.to(device)

# #               # TODO 2 heatmap
# #               heat2 = heat2.to(device)
# #               heat4 = heat4.to(device)

# #               # TODO 3 boundary
# #               boundary1 = boundary1.to(device)

# #               boundary2 = boundary2.to(device)
# #               boundary4 = boundary4.to(device)
# #               boundary8 = boundary8.to(device)
# #               boundary16 = boundary16.to(device)


# #           if args.method == 'Ours':
# #               # ------------------------------------------------------------------------------------#
# #               h, w = args.input_shape[0]//4,args.input_shape[1]//4
# #               _label = F.interpolate(label, size=(h, w), mode='bilinear', align_corners=True)

# #               (feat_out8, feat_out16, feat_out32), \
# #               (feat_out_sp2, feat_out_sp4, feat_out_sp8, feat_out_sp16), \
# #               (pred_loc2, pred_loc4)  = model(image,_label,*(boundary2,boundary4))


# #               #------------------------------------------------------------------------------------#
# #               # TODO 1 seg
# #               #------------------------------------------------------------------------------------#
# #               segloss8 = Dice_loss(feat_out8,label) + nn.CrossEntropyLoss()(feat_out8,png)
# #               segloss16 = Dice_loss(feat_out16,label) + nn.CrossEntropyLoss()(feat_out16,png)
# #               segloss32 = Dice_loss(feat_out32,label) + nn.CrossEntropyLoss()(feat_out32,png)

# #               #--------------------------------------------------------------------------------------#
# #               # TODO add seg
# #               #--------------------------------------------------------------------------------------#
# #               _feat_out_sp2 = up(feat_out_sp2)
# #               _feat_out8 = feat_out8 * (1 - _feat_out_sp2) + feat_out8 + _feat_out_sp2
# #               _seg_loss = Dice_loss(_feat_out8.detach_(),label)


# #               segloss = ( segloss8 + 1.5 * segloss16 + 2.5 * segloss32) + 2. * _seg_loss
# #               #--------------------------------------------------------------------------#
# #               # TODO 2 boundary
# #               #--------------------------------------------------------------------------#
# #               boundaryloss2 = criteror(feat_out_sp2,boundary2)
# #               boundaryloss4 = criteror(feat_out_sp4,boundary4)
# #               boundaryloss8 = criteror(feat_out_sp8,boundary8)
# #               boundaryloss16 = criteror(feat_out_sp16,boundary16)

# #               boundaryloss = (boundaryloss2 + 0.5 * boundaryloss4 + 0.1 * boundaryloss8 + boundaryloss16)

# #               # ------------------------------------------------------------------------------------#
# #               # TODO 3 loc
# #               # ------------------------------------------------------------------------------------#
# #               loc2loss = focal_loss(pred_loc2,heat2)
# #               loc4loss = focal_loss(pred_loc4,heat4)
# #               locloss = (loc2loss + 0.2 * loc4loss)

# #               # TODO KL
# #               #kl = nn.KLDivLoss()(l2,feat_out8)

# #               Tloss = (segloss + boundaryloss + locloss )

# #               # -------------------------------------------------------------------------#
# #               # TODO
# #               # -------------------------------------------------------------------------#
# #               total_cls_loss += (segloss).item()
# #               total_boundary_loss += (boundaryloss).item()
# #               total_heat_loss += (locloss).item()

# #               #total_kl_loss += (kl).item()

# #               total_loss += (total_heat_loss + total_cls_loss + \
# #                              total_boundary_loss + total_kl_loss)

# #           elif (args.method == 'bisenetv1' )|(args.method == 'bisenetv2')\
# #                   |(args.method == 'unet'):

# #                output = model(image)


# #                ds = [] #dice loss
# #                cs = [] #ce loss

# #                for o in output:


# #                    ds.append(Dice_loss(o,label))
# #                    cs.append(nn.CrossEntropyLoss()(o,png))

# #                total_cls_loss += (sum(ds).item() + sum(cs).item())
# #                total_loss += total_cls_loss

# #                Tloss = sum(ds) + sum(cs)

# #           elif (args.method == 'ddrnet' )| (args.method == 'ffnet') | (args.method == 'lpsnet'):

# #               output = model(image)

# #               output = up(output)

# #               loss1 = Dice_loss(output, label)
# #               loss2 = nn.CrossEntropyLoss()(output, png)

# #               total_cls_loss += (loss1 + loss2).item()

# #               total_loss += total_cls_loss

# #               Tloss = (loss1 + loss2)

# #           elif (args.method == 'stdc') | (args.method == 'pidnet'):
# #               ds = [ ]  # dice loss
# #               cs = [ ]  # ce loss

# #               *output,boundary = model(image)

# #               for o in output:
# #                   if args.method == 'pidnet':
# #                      o = up(o)
# #                   ds.append(Dice_loss(o, label))
# #                   cs.append(nn.CrossEntropyLoss()(o, png))

# #               if args.method == 'stdc':
# #                   _boundaryloss = criteror(boundary,boundary8)
# #               else:
# #                   boundary = up(boundary)
# #                   _boundaryloss = criteror(boundary, boundary1)
# #               #stdc_bundaryloss = criteror(boundary,boundary8)
# #               total_boundary_loss += _boundaryloss.item()
# #               total_cls_loss += (sum(ds).item() + sum(cs).item())
# #               total_loss += (total_cls_loss + total_boundary_loss)

# #               Tloss = sum(ds) + sum(cs) + _boundaryloss

# #           Tloss = Tloss / args.accumulate


# #           Tloss.backward()
# #           if ((ind + 1 ) % args.accumulate) == 0:

# #               optimizer.step()

# #               optimizer.zero_grad()


# #           pb.set_postfix(**{
# #               'total_loss': total_loss/(ind + 1),
# #               'total_cls_loss':total_cls_loss/(ind + 1),
# #               'total_boundary_loss':total_boundary_loss/(ind + 1),
# #               'total_heat_loss':total_heat_loss/(ind + 1),
# #               'total_kl_loss':total_kl_loss/(ind + 1)
# #           })

# #           logger.info(f'total_loss: {total_loss/(ind + 1)} total_cls_loss: {total_cls_loss/(ind + 1)}')
# #           logger.info(f'total_boundary_loss: {total_boundary_loss/(ind + 1)}')
# #           logger.info(f'total_heat_loss: {total_heat_loss/(ind + 1)}')


# #           pb.update(1)


# #     #----------------------------------------------------------------------------#
# #     # TODO freeze BN
# #     #----------------------------------------------------------------------------#
# #     model = model.eval()

# #     if (cur_epoch % save_step) == 0:
# #         #Evaluatation
# #         with tqdm(desc = f"Epoch: [{cur_epoch + 1}]/[{total_epoch}]",
# #                   total = len(valdataloader),mininterval = 0.3,postfix = dict,colour = '#7E89EF') as pb:

# #          with torch.no_grad():
# #             # TODO
# #             for ind,batch in enumerate(valdataloader):
# #                 image,label,png,(_,_),((boundary1,boundary2),*boundaryss) = batch

# #                 image = image.to(device)
# #                 boundary2 = boundary2.to(device)
# #                 _b = boundaryss[0].to(device)

# #                 png = png.to(device).detach().cpu().numpy()

# #                 # ------------------------------------------------------------------------------------#
# #                 h, w = args.input_shape[0] // 4, args.input_shape[1] // 4
# #                 _label = F.interpolate(label, size=(h, w), mode='bilinear', align_corners=True)

# #                 if args.method == 'Ours':
# #                     (feat_out8, feat_out16, feat_out32), \
# #                     (feat_out_sp2, feat_out_sp4, feat_out_sp8, feat_out_sp16), \
# #                     (pred_loc2, pred_loc4) = model(image,_label,*(boundary2,_b))

# #                     OneHotlabel8 = F.softmax(feat_out8, dim=1)
# #                     #OneHotlabel16 = F.softmax(feat_out16, dim = 1)
# #                     #OneHotlabel32 = F.softmax(feat_out32, dim = 1)

# #                     targetOneHotlabel8 = torch.argmax((OneHotlabel8), dim=1).detach().cpu().numpy()  # 目标域
# #                     #targetOneHotlabel16 = torch.argmax((OneHotlabel16), dim = 1).detach().cpu().numpy()  # 目标域
# #                     #targetOneHotlabel32 = torch.argmax((OneHotlabel32), dim = 1).detach().cpu().numpy()  # 目标域
# #                     targetlabel = targetOneHotlabel8
# #                     #targetlabel = (targetOneHotlabel8 | targetOneHotlabel16 | targetOneHotlabel32)

# #                 elif (args.method == 'ddrnet') | (args.method == 'ffnet') |(args.method == 'lpsnet'):
# #                     output = model(image)

# #                     pre = F.softmax(up(output), dim = 1)

# #                     targetOneHotlabel8 = torch.argmax(pre, dim = 1).detach().cpu().numpy()

# #                     targetlabel = targetOneHotlabel8
# #                 elif (args.method == 'bisenetv1' )|(args.method == 'bisenetv2' )\
# #                         |(args.method == 'unet'):

# #                     output = model(image)

# #                     pre = F.softmax((output[1]),dim = 1)

# #                     targetOneHotlabel8 = torch.argmax(pre, dim=1).detach().cpu().numpy()

# #                     targetlabel = targetOneHotlabel8

# #                 elif (args.method == 'stdc') | (args.method == 'pidnet'):

# #                     *out,_ = model(image)
# #                     pre = F.softmax(up(out[ 0 ]) if (args.method == 'pidnet') else out[0], dim = 1)

# #                     targetOneHotlabel8 = torch.argmax(pre, dim = 1).detach().cpu().numpy()

# #                     targetlabel = targetOneHotlabel8

# #                 eval.init(png,targetlabel)

# #                 pb.update(1)

# #             # 评估结果
# #             eval.show()

# #             res = eval.get_res()

# #             logger.info(f'Epoch: {cur_epoch}  res: {res}')

# #     if ((cur_epoch + 1)%save_step == 0) :
# #             torch.save(model.module.state_dict(),f'{save_path}/{(cur_epoch + 1)}.pth')
# #     if (cur_epoch + 1) == total_epoch:
# #      torch.save(model.module.state_dict(),f'{save_path}/last.pth')


# from tqdm import tqdm
# import torch
# import numpy as np
# from nets.criterion import Dice_loss,score,\
#      Heatmaploss,focal_loss
# import torch.nn as nn
# from utils.tools import get_lr
# from utils.tools import prob2entropy
# import torch.nn.functional as F
# from utils.config import Config


# def epoch_fit(cur_epoch,total_epoch,save_step,
#               model,optimizer,dataloader,
#               device,logger,args,save_path,
#               eval,local_rank): #评估函数是对测试集


#     model = model.to(device)

#     if torch.cuda.device_count() > 1:
#         #-----------------------------------------------------#
#         #                  训练的BS要能被显卡数整除
#         #-----------------------------------------------------#
#         #print(f'GPUS Numbers :{torch.cuda.device_count()}')

#         #-------------------------------------------------------#
#         #                         设置多卡训练
#         #-------------------------------------------------------#
#         model = nn.DataParallel(model)


#     traindataloader,testdataloader,valdataloader = dataloader


#     # 损失函数定义
#     criteror = nn.BCEWithLogitsLoss()
#     heatloss = Heatmaploss() # TODO ? loc loss ?


#     # 上采样
#     up = nn.Upsample(size = args.input_shape,
#                      mode = 'bilinear', align_corners = True)

#     with tqdm(desc = f"Epoch: [{cur_epoch + 1}]/[{total_epoch}]",
#               total = len(traindataloader),mininterval = 0.3,postfix = dict,colour = '#6DB9EF') as pb:

#         model = model.train()

#         #-------------------------------------#
#         # 训练损失计算
#         #-------------------------------------#
#         total_loss = 0.
#         total_cls_loss = 0.
#         total_boundary_loss = 0.
#         total_heat_loss = 0.
#         total_kl_loss = 0.

#         # 源域
#         for ind,batch in enumerate(traindataloader):
#           image,label,png,(heat2,heat4),((boundary1,boundary2),boundary4,boundary8,boundary16) = batch
#           with torch.no_grad():
#               # TODO 1
#               image = image.to(device)
#               label = label.to(device)
#               png = png.to(device)

#               # TODO 2 heatmap
#               heat2 = heat2.to(device)
#               heat4 = heat4.to(device)

#               # TODO 3 boundary
#               boundary1 = boundary1.to(device)

#               boundary2 = boundary2.to(device)
#               boundary4 = boundary4.to(device)
#               boundary8 = boundary8.to(device)
#               boundary16 = boundary16.to(device)


#           if args.method == 'Ours':
#               # ------------------------------------------------------------------------------------#
#               h, w = args.input_shape[0]//4,args.input_shape[1]//4
#               _label = F.interpolate(label, size=(h, w), mode='bilinear', align_corners=True)

#               (feat_out8, feat_out16, feat_out32), \
#               (feat_out_sp2, feat_out_sp4, feat_out_sp8, feat_out_sp16), \
#               (pred_loc2, pred_loc4)  = model(image,_label,*(boundary2,boundary4))


#               #------------------------------------------------------------------------------------#
#               # TODO 1 seg
#               #------------------------------------------------------------------------------------#
#               segloss8 = Dice_loss(feat_out8,label) + nn.CrossEntropyLoss()(feat_out8,png)

#             #   pos = (feat_out8 * label).float()
#             #   neg = (feat_out8 * (1 - label)).float()

#             #   _segloss = nn.CosineSimilarity()(pos,label).mean() + 0.5 * nn.CosineSimilarity()(neg,label).mean()
#               #_segloss = criteror(feat_out8,png.unsqueeze(dim=1))

#               segloss16 = Dice_loss(feat_out16,label) + nn.CrossEntropyLoss()(feat_out16,png)
#               segloss32 = Dice_loss(feat_out32,label) + nn.CrossEntropyLoss()(feat_out32,png)

#               #--------------------------------------------------------------------------------------#
#               # TODO add seg
#               #--------------------------------------------------------------------------------------#
#               # _feat_out_sp2 = up(feat_out_sp2)
#               # _feat_out8 = feat_out8 * (1 - _feat_out_sp2) + feat_out8 + _feat_out_sp2
#               # _seg_loss = nn.KLDivLoss()(_feat_out8,label)

#               # Kl
#               _feat = up(feat_out_sp4)
#               _kl_s = nn.MSELoss()(up(feat_out_sp2),_feat)


#               # TODO 230
#               #with torch.no_grad():
#               alpha = (feat_out8 - label).mean()
#               alpha = alpha ** 2

#             #   alpha1 = (feat_out16 - label).mean()
#             #   alpha1 = alpha1 ** 2

#                   # alpha2 = (feat_out32 - label).mean()
#                   # alpha2 = alpha2 ** 2


#               segloss = ( alpha *  segloss8 +   segloss16  + segloss32  )
#               #--------------------------------------------------------------------------#
#               # TODO 2 boundary
#               #--------------------------------------------------------------------------#
#               # boundaryloss2 = nn.KLDivLoss()((feat_out_sp2),boundary2) + nn.MSELoss()((feat_out_sp2),boundary2)
#               # feat_out_sp1x = up(feat_out_sp2)
#               # pos = (feat_out_sp1x * boundary1).float()
#               # neg = (feat_out_sp1x * (1 - boundary1)).float()
#               #boundaryloss2 = nn.CosineSimilarity()(pos,boundary1).mean() + 0.5 * nn.CosineSimilarity()(neg,boundary1).mean()
#               #boundaryloss2 = criteror(feat_out_sp2,boundary1)
#             #   if cur_epoch >= 170:
#             #    with torch.no_grad(): # 170 开始
#             #     bate = (feat_out_sp2 - boundary2).mean() **2
#             #   else:
#               bate = (feat_out_sp2 - boundary2).mean() **2

#               boundaryloss2 = heatloss(feat_out_sp2,boundary2)

#               # TODO 230 之后将 heatloss -> BCE
#               boundaryloss4 = heatloss(feat_out_sp4,boundary4)

#               # TODO 230
#               boundaryloss8 = criteror(feat_out_sp8,boundary8)

#               boundaryloss = (bate * boundaryloss2 + boundaryloss4  + boundaryloss8)

#               # ------------------------------------------------------------------------------------#
#               # TODO 3 loc
#               # ------------------------------------------------------------------------------------#
#               loc2loss = focal_loss(pred_loc2,heat2)
#               loc4loss = focal_loss(pred_loc4,heat4)
#               locloss = (loc2loss + 0.2 * loc4loss)

#               # TODO KL
#               #kl = nn.KLDivLoss()(l2,feat_out8)

#               Tloss = ( boundaryloss + segloss  + _kl_s )  #目前就只做一个边界

#               # -------------------------------------------------------------------------#
#               # TODO
#               # -------------------------------------------------------------------------#
#               total_cls_loss += (segloss).item()
#               total_boundary_loss += (boundaryloss).item()
#               total_heat_loss += (locloss).item()

#               total_kl_loss += (_kl_s).item()

#               total_loss += (total_heat_loss + total_cls_loss + \
#                              total_boundary_loss + total_kl_loss)

#           elif (args.method == 'bisenetv1' )|(args.method == 'bisenetv2')\
#                   |(args.method == 'unet'):

#                output = model(image)


#                ds = [] #dice loss
#                cs = [] #ce loss

#                for o in output:


#                    ds.append(Dice_loss(o,label))
#                    cs.append(nn.CrossEntropyLoss()(o,png))

#                total_cls_loss += (sum(ds).item() + sum(cs).item())
#                total_loss += total_cls_loss

#                Tloss = sum(ds) + sum(cs)

#           elif (args.method == 'ddrnet' )| (args.method == 'ffnet') | (args.method == 'lpsnet'):

#               output = model(image)

#               output = up(output)

#               loss1 = Dice_loss(output, label)
#               loss2 = nn.CrossEntropyLoss()(output, png)

#               total_cls_loss += (loss1 + loss2).item()

#               total_loss += total_cls_loss

#               Tloss = (loss1 + loss2)

#           elif (args.method == 'stdc') | (args.method == 'pidnet'):
#               ds = [ ]  # dice loss
#               cs = [ ]  # ce loss

#               *output,boundary = model(image)

#               for o in output:
#                   if args.method == 'pidnet':
#                      o = up(o)
#                   ds.append(Dice_loss(o, label))
#                   cs.append(nn.CrossEntropyLoss()(o, png))

#               if args.method == 'stdc':
#                   _boundaryloss = criteror(boundary,boundary8)
#               else:
#                   boundary = up(boundary)
#                   _boundaryloss = criteror(boundary, boundary1)
#               #stdc_bundaryloss = criteror(boundary,boundary8)
#               total_boundary_loss += _boundaryloss.item()
#               total_cls_loss += (sum(ds).item() + sum(cs).item())
#               total_loss += (total_cls_loss + total_boundary_loss)

#               Tloss = sum(ds) + sum(cs) + _boundaryloss

#           Tloss = Tloss / args.accumulate


#           Tloss.backward()
#           if ((ind + 1 ) % args.accumulate) == 0:

#               optimizer.step()

#               optimizer.zero_grad()


#           pb.set_postfix(**{
#               'total_loss': total_loss/(ind + 1),
#               'total_cls_loss':total_cls_loss/(ind + 1),
#               'total_boundary_loss':total_boundary_loss/(ind + 1),
#               'total_heat_loss':total_heat_loss/(ind + 1),
#               'total_kl_loss':total_kl_loss/(ind + 1)
#           })

#           logger.info(f'total_loss: {total_loss/(ind + 1)} total_cls_loss: {total_cls_loss/(ind + 1)}')
#           logger.info(f'total_boundary_loss: {total_boundary_loss/(ind + 1)}')
#           logger.info(f'total_heat_loss: {total_heat_loss/(ind + 1)}')


#           pb.update(1)


#     #----------------------------------------------------------------------------#
#     # TODO freeze BN
#     #----------------------------------------------------------------------------#
#     model = model.eval()

#     if (cur_epoch % save_step) == 0:
#         #Evaluatation
#         with tqdm(desc = f"Epoch: [{cur_epoch + 1}]/[{total_epoch}]",
#                   total = len(valdataloader),mininterval = 0.3,postfix = dict,colour = '#7E89EF') as pb:

#          with torch.no_grad():
#             # TODO
#             for ind,batch in enumerate(valdataloader):
#                 image,label,png,(_,_),((boundary1,boundary2),*boundaryss) = batch

#                 image = image.to(device)
#                 boundary2 = boundary2.to(device)
#                 boundary2 = torch.randn_like(boundary2)
#                 _b = boundaryss[0].to(device)
#                 _b = torch.randn_like(_b)

#                 png = png.to(device).detach().cpu().numpy()

#                 # ------------------------------------------------------------------------------------#
#                 h, w = args.input_shape[0] // 4, args.input_shape[1] // 4
#                 _label = F.interpolate(label, size=(h, w), mode='bilinear', align_corners=True)

#                 if args.method == 'Ours':
#                     (feat_out8, feat_out16, feat_out32), \
#                     (feat_out_sp2, feat_out_sp4, feat_out_sp8, feat_out_sp16), \
#                     (pred_loc2, pred_loc4) = model(image,_label,*(boundary2,_b))

#                     OneHotlabel8 = F.softmax(feat_out8, dim=1)
#                     #OneHotlabel16 = F.softmax(feat_out16, dim = 1)
#                     #OneHotlabel32 = F.softmax(feat_out32, dim = 1)

#                     targetOneHotlabel8 = torch.argmax((OneHotlabel8), dim=1).detach().cpu().numpy()  # 目标域
#                     #targetOneHotlabel16 = torch.argmax((OneHotlabel16), dim = 1).detach().cpu().numpy()  # 目标域
#                     #targetOneHotlabel32 = torch.argmax((OneHotlabel32), dim = 1).detach().cpu().numpy()  # 目标域
#                     targetlabel = targetOneHotlabel8
#                     #targetlabel = (targetOneHotlabel8 | targetOneHotlabel16 | targetOneHotlabel32)

#                 elif (args.method == 'ddrnet') | (args.method == 'ffnet') |(args.method == 'lpsnet'):
#                     output = model(image)

#                     pre = F.softmax(up(output), dim = 1)

#                     targetOneHotlabel8 = torch.argmax(pre, dim = 1).detach().cpu().numpy()

#                     targetlabel = targetOneHotlabel8
#                 elif (args.method == 'bisenetv1' )|(args.method == 'bisenetv2' )\
#                         |(args.method == 'unet'):

#                     output = model(image)

#                     pre = F.softmax((output[1]),dim = 1)

#                     targetOneHotlabel8 = torch.argmax(pre, dim=1).detach().cpu().numpy()

#                     targetlabel = targetOneHotlabel8

#                 elif (args.method == 'stdc') | (args.method == 'pidnet'):

#                     *out,_ = model(image)
#                     pre = F.softmax(up(out[ 0 ]) if (args.method == 'pidnet') else out[0], dim = 1)

#                     targetOneHotlabel8 = torch.argmax(pre, dim = 1).detach().cpu().numpy()

#                     targetlabel = targetOneHotlabel8

#                 eval.init(png,targetlabel)

#                 pb.update(1)

#             # 评估结果
#             eval.show()

#             res = eval.get_res()

#             logger.info(f'Epoch: {cur_epoch}  res: {res}')

#     if ((cur_epoch + 1)%save_step == 0) :
#             torch.save(model.module.state_dict(),f'{save_path}/{(cur_epoch + 1)}.pth')
#     if (cur_epoch + 1) == total_epoch:
#      torch.save(model.module.state_dict(),f'{save_path}/last.pth')


from tqdm import tqdm
import torch
import numpy as np
from nets.criterion import Dice_loss, score, \
    Heatmaploss, focal_loss
import torch.nn as nn
from utils.tools import get_lr
from utils.tools import prob2entropy
import torch.nn.functional as F
from utils.config import Config


def epoch_fit(cur_epoch, total_epoch, save_step,
              model, optimizer, dataloader,
              device, logger, args, save_path,
              eval, local_rank):  # 评估函数是对测试集

    model = model.to(device)

    if torch.cuda.device_count() > 1:
        # -----------------------------------------------------#
        #                  训练的BS要能被显卡数整除
        # -----------------------------------------------------#
        # print(f'GPUS Numbers :{torch.cuda.device_count()}')

        # -------------------------------------------------------#
        #                         设置多卡训练
        # -------------------------------------------------------#
        model = nn.DataParallel(model)

    traindataloader, testdataloader, valdataloader = dataloader

    # 损失函数定义
    criteror = nn.BCEWithLogitsLoss()
    heatloss = Heatmaploss(alpha = args.B)  # TODO ? loc loss ?

    # 上采样
    up = nn.Upsample(size = args.input_shape,
                     mode = 'bilinear', align_corners = True)

    with tqdm(desc = f"Epoch: [{cur_epoch + 1}]/[{total_epoch}]",
              total = len(traindataloader), mininterval = 0.3, postfix = dict, colour = '#6DB9EF') as pb:

        model = model.train()

        # -------------------------------------#
        # 训练损失计算
        # -------------------------------------#
        total_loss = 0.
        total_cls_loss = 0.
        total_boundary_loss = 0.
        total_heat_loss = 0.
        total_kl_loss = 0.

        # 源域
        for ind, batch in enumerate(traindataloader):
            image, label, png, (heat2, heat4), ((boundary1, boundary2), boundary4, boundary8, boundary16) = batch
            with torch.no_grad():
                # TODO 1
                image = image.to(device)
                label = label.to(device)
                png = png.to(device)

                # TODO 2 heatmap
                heat2 = heat2.to(device)
                heat4 = heat4.to(device)

                # TODO 3 boundary
                boundary1 = boundary1.to(device)

                boundary2 = boundary2.to(device)
                boundary4 = boundary4.to(device)
                boundary8 = boundary8.to(device)
                boundary16 = boundary16.to(device)

            if args.method == 'Ours':
                # ------------------------------------------------------------------------------------#
                h, w = args.input_shape[ 0 ] // 4, args.input_shape[ 1 ] // 4
                _label = F.interpolate(label, size = (h, w), mode = 'bilinear', align_corners = True)

                (feat_out8, feat_out16, feat_out32), \
                (feat_out_sp2, feat_out_sp4, feat_out_sp8, feat_out_sp16), \
                (pred_loc2, pred_loc4) = model(image) #model(image, _label, *(boundary2, boundary4))

                # ------------------------------------------------------------------------------------#
                # TODO 1 seg
                # ------------------------------------------------------------------------------------#
                segloss8 = Dice_loss(feat_out8, label) + nn.CrossEntropyLoss()(feat_out8, png)

                #   pos = (feat_out8 * label).float()
                #   neg = (feat_out8 * (1 - label)).float()

                #   _segloss = nn.CosineSimilarity()(pos,label).mean() + 0.5 * nn.CosineSimilarity()(neg,label).mean()
                # _segloss = criteror(feat_out8,png.unsqueeze(dim=1))

                segloss16 = Dice_loss(feat_out16, label) + nn.CrossEntropyLoss()(feat_out16, png)
                segloss32 = Dice_loss(feat_out32, label) + nn.CrossEntropyLoss()(feat_out32, png)

                # --------------------------------------------------------------------------------------#
                # TODO add seg
                # --------------------------------------------------------------------------------------#
                # _feat_out_sp2 = up(feat_out_sp2)
                # _feat_out8 = feat_out8 * (1 - _feat_out_sp2) + feat_out8 + _feat_out_sp2
                # _seg_loss = nn.KLDivLoss()(_feat_out8,label)

                if args.boundary_dis:
                    # Kl
                    # 边界自蒸馏损失
                    _feat = up(feat_out_sp4)
                    _kl_s = nn.MSELoss()(F.sigmoid(up(feat_out_sp2)), F.sigmoid(_feat))
                else:
                    _kl_s = 0.

                if args.self_segdis:
                    # 像素对比
                    # 随着模型的拟合，负样本的会不断偏向于0
                    _seg_kl1 = nn.KLDivLoss()(F.log_softmax(feat_out16, dim = 1), F.softmax(feat_out8, dim = 1))
                    _seg_kl2 = nn.KLDivLoss()(F.log_softmax(feat_out8, dim = 1), F.softmax(feat_out16, dim = 1))
                    _seg_kl = (_seg_kl1 + _seg_kl2) * args.A
                else:
                    _seg_kl = 0.
                # 混合交叉损失

                if ((cur_epoch & 1) == 0):
                    alpha = (feat_out8 - label).mean()
                    alpha = alpha ** 2
                else:
                    # TODO
                    with torch.no_grad():
                        alpha = (feat_out8 - label).mean()
                        alpha = alpha ** 2

                #          alpha1 = (feat_out16 - feat_out32).mean() ** 2

                #   segloss = ( alpha *  segloss8 +  segloss16 +  segloss32 )
                if ((cur_epoch & 1) == 0):
                    segloss = (alpha * segloss8 + (segloss16 + segloss32) + _seg_kl)
                else:
                    segloss = (alpha * segloss8 + (segloss16 + segloss32))
                # --------------------------------------------------------------------------#
                # TODO 2 boundary
                # --------------------------------------------------------------------------#
                # boundaryloss2 = nn.KLDivLoss()((feat_out_sp2),boundary2) + nn.MSELoss()((feat_out_sp2),boundary2)
                # feat_out_sp1x = up(feat_out_sp2)
                # pos = (feat_out_sp1x * boundary1).float()
                # neg = (feat_out_sp1x * (1 - boundary1)).float()
                # boundaryloss2 = nn.CosineSimilarity()(pos,boundary1).mean() + 0.5 * nn.CosineSimilarity()(neg,boundary1).mean()
                # boundaryloss2 = criteror(feat_out_sp2,boundary1)
                # with torch.no_grad(): # TODO 掉精度
                if ((cur_epoch & 1) != 0):
                    bate = (feat_out_sp2 - boundary2).mean() ** 2
                else:
                    with torch.no_grad():
                        bate = (feat_out_sp2 - boundary2).mean() ** 2
                if args.boundary:
                    boundaryloss2 = heatloss(feat_out_sp2, boundary2)
                    boundaryloss4 = heatloss(feat_out_sp4, boundary4)
                    boundaryloss8 = criteror(feat_out_sp8, boundary8)

                    boundaryloss = (bate * boundaryloss2 + boundaryloss4 + boundaryloss8)
                else:
                    boundaryloss = 0.
                # ------------------------------------------------------------------------------------#
                # TODO 3 loc
                # ------------------------------------------------------------------------------------#
                if args.loc_loss:
                    loc2loss = focal_loss(pred_loc2, heat2,args.alpha,args.bate)
                    loc4loss = focal_loss(pred_loc4, heat4,args.alpha,args.bate)
                    locloss = (loc2loss + args.L * loc4loss)
                else:
                    locloss = 0.
                # TODO KL
                # kl = nn.KLDivLoss()(l2,feat_out8)
                # ------------------------------------------------------------#
                # step1. 没有loc
                # ------------------------------------------------------------$

                Tloss = (boundaryloss + segloss + _kl_s + locloss)

                # -------------------------------------------------------------------------#
                # TODO
                # -------------------------------------------------------------------------#
                total_cls_loss += (segloss).item()
                total_boundary_loss += (boundaryloss).item() if args.boundary else 0.
                total_heat_loss += (locloss).item() if args.loc_loss else 0.

                total_kl_loss += (_kl_s).item() if args.boundary_dis else 0.

                total_loss += (total_heat_loss + total_cls_loss + \
                               total_boundary_loss + total_kl_loss)

            elif (args.method == 'bisenetv1') | (args.method == 'bisenetv2') \
                    | (args.method == 'unet') | (args.method == 'ablation'):

                output = model(image)

                ds = [ ]  # dice loss
                cs = [ ]  # ce loss

                for o in output:
                    ds.append(Dice_loss(o, label))
                    cs.append(nn.CrossEntropyLoss()(o, png))

                total_cls_loss += (sum(ds).item() + sum(cs).item())
                total_loss += total_cls_loss

                Tloss = sum(ds) + sum(cs)

            elif (args.method == 'ddrnet') | (args.method == 'ffnet') | (args.method == 'lpsnet'):

                output = model(image)

                output = up(output)

                loss1 = Dice_loss(output, label)
                loss2 = nn.CrossEntropyLoss()(output, png)

                total_cls_loss += (loss1 + loss2).item()

                total_loss += total_cls_loss

                Tloss = (loss1 + loss2)

            elif (args.method == 'stdc') | (args.method == 'pidnet'):
                ds = [ ]  # dice loss
                cs = [ ]  # ce loss

                *output, boundary = model(image)

                for o in output:
                    if args.method == 'pidnet':
                        o = up(o)
                    ds.append(Dice_loss(o, label))
                    cs.append(nn.CrossEntropyLoss()(o, png))

                if args.method == 'stdc':
                    _boundaryloss = criteror(boundary, boundary8)
                else:
                    boundary = up(boundary)
                    _boundaryloss = criteror(boundary, boundary1)
                # stdc_bundaryloss = criteror(boundary,boundary8)
                total_boundary_loss += _boundaryloss.item()
                total_cls_loss += (sum(ds).item() + sum(cs).item())
                total_loss += (total_cls_loss + total_boundary_loss)

                Tloss = sum(ds) + sum(cs) + _boundaryloss

            Tloss = Tloss / args.accumulate

            Tloss.backward()
            if ((ind + 1) % args.accumulate) == 0:
                optimizer.step()

                optimizer.zero_grad()

            pb.set_postfix(**{
                'total_loss': total_loss / (ind + 1),
                'total_cls_loss': total_cls_loss / (ind + 1),
                'total_boundary_loss': total_boundary_loss / (ind + 1),
                'total_heat_loss': total_heat_loss / (ind + 1),
                'total_kl_loss': total_kl_loss / (ind + 1)
            })

            logger.info(f'total_loss: {total_loss / (ind + 1)} total_cls_loss: {total_cls_loss / (ind + 1)}')
            logger.info(f'total_boundary_loss: {total_boundary_loss / (ind + 1)}')
            logger.info(f'total_heat_loss: {total_heat_loss / (ind + 1)}')

            pb.update(1)

    # ----------------------------------------------------------------------------#
    # TODO freeze BN
    # ----------------------------------------------------------------------------#
    model = model.eval()

    if (cur_epoch % save_step) == 0:
        # Evaluatation
        with tqdm(desc = f"Epoch: [{cur_epoch + 1}]/[{total_epoch}]",
                  total = len(valdataloader), mininterval = 0.3, postfix = dict, colour = '#7E89EF') as pb:

            with torch.no_grad():
                # TODO
                for ind, batch in enumerate(valdataloader):
                    image, label, png, (_, _), ((boundary1, boundary2), *boundaryss) = batch

                    image = image.to(device)
                    boundary2 = boundary2.to(device)
                    boundary2 = torch.randn_like(boundary2)
                    _b = boundaryss[ 0 ].to(device)
                    _b = torch.randn_like(_b)

                    png = png.to(device).detach().cpu().numpy()

                    # ------------------------------------------------------------------------------------#
                    h, w = args.input_shape[ 0 ] // 4, args.input_shape[ 1 ] // 4
                    _label = F.interpolate(label, size = (h, w), mode = 'bilinear', align_corners = True)

                    if args.method == 'Ours':
                        (feat_out8, feat_out16, feat_out32), \
                        (feat_out_sp2, feat_out_sp4, feat_out_sp8, feat_out_sp16), \
                        (pred_loc2, pred_loc4) = model(image, _label, *(boundary2, _b))

                        OneHotlabel8 = F.softmax(feat_out8, dim = 1)
                        # OneHotlabel16 = F.softmax(feat_out16, dim = 1)
                        # OneHotlabel32 = F.softmax(feat_out32, dim = 1)

                        targetOneHotlabel8 = torch.argmax((OneHotlabel8), dim = 1).detach().cpu().numpy()  # 目标域
                        # targetOneHotlabel16 = torch.argmax((OneHotlabel16), dim = 1).detach().cpu().numpy()  # 目标域
                        # targetOneHotlabel32 = torch.argmax((OneHotlabel32), dim = 1).detach().cpu().numpy()  # 目标域
                        targetlabel = targetOneHotlabel8
                        # targetlabel = (targetOneHotlabel8 | targetOneHotlabel16 | targetOneHotlabel32)

                    elif (args.method == 'ddrnet') | (args.method == 'ffnet') | (args.method == 'lpsnet'):
                        output = model(image)

                        pre = F.softmax(up(output), dim = 1)

                        targetOneHotlabel8 = torch.argmax(pre, dim = 1).detach().cpu().numpy()

                        targetlabel = targetOneHotlabel8
                    elif (args.method == 'bisenetv1') | (args.method == 'bisenetv2') \
                            | (args.method == 'unet') | (args.method == 'ablation'):

                        output = model(image)

                        pre = F.softmax((output[ 1 ]), dim = 1)

                        targetOneHotlabel8 = torch.argmax(pre, dim = 1).detach().cpu().numpy()

                        targetlabel = targetOneHotlabel8

                    elif (args.method == 'stdc') | (args.method == 'pidnet'):

                        *out, _ = model(image)
                        pre = F.softmax(up(out[ 0 ]) if (args.method == 'pidnet') else out[ 0 ], dim = 1)

                        targetOneHotlabel8 = torch.argmax(pre, dim = 1).detach().cpu().numpy()

                        targetlabel = targetOneHotlabel8

                    eval.init(png, targetlabel)

                    pb.update(1)

                # 评估结果
                eval.show()

                res = eval.get_res()

                logger.info(f'Epoch: {cur_epoch}  res: {res}')

    if ((cur_epoch + 1) % save_step == 0):
        torch.save(model.module.state_dict(), f'{save_path}/{(cur_epoch + 1)}.pth')
    if (cur_epoch + 1) == total_epoch:
        torch.save(model.module.state_dict(), f'{save_path}/last.pth')
