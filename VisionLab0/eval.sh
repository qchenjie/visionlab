#----------------------------------------------#
# TODO Ours
#----------------------------------------------#
# python eval.py --method Ours --model_path Pth\\Golbal_city\\Ours\\50.pth \
#        --val_file M_val.txt


#++++++++++++++++++++++++++++++++++++++++++++++++#
# TODO STDC
#++++++++++++++++++++++++++++++++++++++++++++++++#
# python eval.py --method stdc --model_path Pth\\Golbal_city\\stdc\\180.pth \
#        --val_file Global_city_val.txt

# python eval.py --method lpsnet --model_path No_data\\Pth\\lpsnet\\last.pth \
#        --val_file val.txt

# python eval.py --method stdc --model_path No_data\\Pth\\stdc\\last.pth \
#        --val_file val.txt


# python eval.py --method pidnet --model_path No_data\\Pth\\pidnet\\last.pth \
#        --val_file val.txt

#python eval.py --method bisenetv1 --model_path No_data\\Pth\\bisenetv1\\last.pth \
#       --val_file val.txt


# Unet
#python eval.py --method ddrnet --model_path Pth\\Mas\\ddrnet\\60.pth \
#       --val_file M_val.txt --datasetname Mas

#python eval.py --method Ours --model_path Pth\\Global_city_Guidence\\Ours\\2400.pth \
#       --val_file Global_city_val.txt --datasetname Global_city --h 1024 --w 1024


#python eval.py --method Ours --model_path Pth\\Aerial_Guidence\\Ours\\70.pth \
#       --val_file val.txt --datasetname Aerial_Guidence --h 1024 --w 1024


#python eval.py --method Ours --model_path Pth\\7_25_Mas_loc_loss\\Ours\\300.pth \
#       --val_file M_val.txt --datasetname XXX --h 512 --w 512


#---------------------------------------------------------------------------------#
#  对比方法 Mas 数据集
#---------------------------------------------------------------------------------#
# bisenetv1
#python eval.py --method bisenetv1 --model_path Pth\\Ubuntu\\Pth\\Mas\\bisenetv1\\last.pth \
#       --val_file _M_val.txt --datasetname Mas --h 512 --w 512
## bisenetv2
#python eval.py --method bisenetv2 --model_path Pth\\Mas\\bisenetv2\\last.pth \
#       --val_file _M_val.txt --datasetname Mas --h 512 --w 512
#
## ddrnet
#python eval.py --method ddrnet --model_path Pth\\Mas\\ddrnet\\last.pth \
#       --val_file _M_val.txt --datasetname Mas --h 512 --w 512
#
## ffnet
#python eval.py --method ffnet --model_path Pth\\Ubuntu\\Pth\\Mas\\ffnet\\last.pth \
#       --val_file _M_val.txt --datasetname Mas --h 512 --w 512

# pidnet
#python eval.py --method pidnet --model_path Pth\\Mas\\pidnet\\270.pth \
#       --val_file _M_val.txt --datasetname Mas --h 512 --w 512
# stdc
#python eval.py --method stdc --model_path Pth\\Mas\\stdc\\last.pth \
#       --val_file _M_val.txt --datasetname Mas --h 512 --w 512

# lspnet
#python eval.py --method lpsnet --model_path Pth\\Ubuntu\\Pth\\Mas\\lpsnet\\last.pth \
#       --val_file _M_val.txt --datasetname Mas --h 512 --w 512



#---------------------------------------------------------------------------------#
#  对比方法 Aerial 数据集
#---------------------------------------------------------------------------------#
# bisenetv1
#python eval.py --method bisenetv1 --model_path Pth\\Ubuntu\\Pth\\Aerial\\bisenetv1\\last.pth \
#       --val_file _Aerial_val.txt --datasetname Aerial --h 512 --w 512
### bisenetv2
#python eval.py --method bisenetv2 --model_path Pth\\Aerial\\bisenetv2\\last.pth \
#       --val_file _Aerial_val.txt --datasetname Aerial --h 512 --w 512
##
### ddrnet
#python eval.py --method ddrnet --model_path  Pth\\Ubuntu\\Pth\\Aerial\\ddrnet\\last.pth \
#       --val_file _Aerial_val.txt --datasetname Aerial --h 512 --w 512
##
### ffnet
#python eval.py --method ffnet --model_path Pth\\Ubuntu\\Pth\\Aerial\\ffnet\\last.pth \
#       --val_file _Aerial_val.txt --datasetname Aerial --h 512 --w 512
#
## pidnet
#python eval.py --method pidnet --model_path  Pth\\Ubuntu\\Pth\\Aerial\\pidnet\\last.pth \
#       --val_file _Aerial_val.txt --datasetname Aerial --h 512 --w 512
#
## stdc
#python eval.py --method stdc --model_path  Pth\\Ubuntu\\Pth\\Aerial\\stdc\\last.pth \
#       --val_file _Aerial_val.txt --datasetname Aerial --h 512 --w 512
#
## lspnet
#python eval.py --method lpsnet --model_path Pth\\Ubuntu\\Pth\\Aerial\\lpsnet\\last.pth \
#       --val_file _Aerial_val.txt --datasetname 8_11_Test --h 512 --w 512

# OursPth\Ubuntu\Pth\7_29_Mas_loc_loss_4\Ours
#python eval.py --method Ours --model_path Pth\\Ubuntu\\Pth\\7_29_Mas_loc_loss_4\\Ours\\last.pth \
#       --val_file _M_val.txt --datasetname 8_11_Test --h 512 --w 512

#---------------------------------------------------------------------------------#
#  交叉验证模型 robust Mas训练预测global
#---------------------------------------------------------------------------------#
# bisenetv1
#python eval.py --method bisenetv1 --model_path Pth\\Ubuntu\\Pth\\Mas\\bisenetv1\\last.pth \
#       --val_file _Global_val.txt --datasetname Mas_global --h 512 --w 512
### bisenetv2
#python eval.py --method bisenetv2 --model_path Pth\\Mas\\bisenetv2\\last.pth \
#       --val_file _Global_val.txt --datasetname Mas_global --h 512 --w 512
##
### ddrnet
#python eval.py --method ddrnet --model_path Pth\\Mas\\ddrnet\\last.pth \
#       --val_file _Global_val.txt --datasetname Mas_global --h 512 --w 512
##
### ffnet
#python eval.py --method ffnet --model_path Pth\\Ubuntu\\Pth\\Mas\\ffnet\\last.pth \
#       --val_file _Global_val.txt --datasetname Mas_global --h 512 --w 512
#
## pidnet
#python eval.py --method pidnet --model_path Pth\\Mas\\pidnet\\270.pth \
#       --val_file _Global_val.txt --datasetname Mas_global --h 512 --w 512
## stdc
#python eval.py --method stdc --model_path Pth\\Mas\\stdc\\last.pth \
#       --val_file _Global_val.txt --datasetname Mas_global --h 512 --w 512
#
## lspnet
#python eval.py --method lpsnet --model_path Pth\\Ubuntu\\Pth\\Mas\\lpsnet\\last.pth \
#       --val_file _Global_val.txt --datasetname Mas_global --h 512 --w 512
#
#
#
##---------------------------------------------------------------------------------#
##  交叉验证 Aerial训练 Global预测
##---------------------------------------------------------------------------------#
## bisenetv1
#python eval.py --method bisenetv1 --model_path Pth\\Ubuntu\\Pth\\Aerial\\bisenetv1\\last.pth \
#       --val_file _Global_val.txt --datasetname Aerial_global --h 512 --w 512
### bisenetv2
#python eval.py --method bisenetv2 --model_path Pth\\Aerial\\bisenetv2\\last.pth \
#       --val_file _Global_val.txt --datasetname Aerial_global --h 512 --w 512
##
### ddrnet
#python eval.py --method ddrnet --model_path  Pth\\Ubuntu\\Pth\\Aerial\\ddrnet\\last.pth \
#       --val_file _Global_val.txt --datasetname Aerial_global --h 512 --w 512
##
### ffnet
#python eval.py --method ffnet --model_path Pth\\Ubuntu\\Pth\\Aerial\\ffnet\\last.pth \
#       --val_file _Global_val.txt --datasetname Aerial_global --h 512 --w 512
#
## pidnet
#python eval.py --method pidnet --model_path  Pth\\Ubuntu\\Pth\\Aerial\\pidnet\\last.pth \
#       --val_file _Global_val.txt --datasetname Aerial_global --h 512 --w 512
#
### stdc
#python eval.py --method stdc --model_path  Pth\\Ubuntu\\Pth\\Aerial\\stdc\\last.pth \
#       --val_file _Global_val.txt --datasetname Aerial_global --h 512 --w 512
##
### lspnet
#python eval.py --method lpsnet --model_path Pth\\Ubuntu\\Pth\\Aerial\\lpsnet\\last.pth \
#       --val_file _Global_val.txt --datasetname Aerial_global --h 512 --w 512

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#-----------------------------------------------------------------------#
#   消融实验: 参数消融
#-----------------------------------------------------------------------#
## Ours
#python eval.py --method Ours --model_path Pth\\8_8_Aerial_Par1\\Ours\\last.pth \
#       --val_file _Aerial_val.txt --datasetname Aerial_Par1 --h 512 --w 512
#
#python eval.py --method Ours --model_path Pth\\8_8_Aerial_Par2\\Ours\\last.pth \
#       --val_file _Aerial_val.txt --datasetname Aerial_Par2 --h 512 --w 512
#
#python eval.py --method Ours --model_path Pth\\8_8_Aerial_Par3\\Ours\\last.pth \
#       --val_file _Aerial_val.txt --datasetname Aerial_Par3 --h 512 --w 512
#
#python eval.py --method Ours --model_path Pth\\8_8_Aerial_Par4\\Ours\\last.pth \
#       --val_file _Aerial_val.txt --datasetname Aerial_Par4 --h 512 --w 512
#
#python eval.py --method Ours --model_path Pth\\8_8_Aerial_Par5\\Ours\\last.pth \
#       --val_file _Aerial_val.txt --datasetname Aerial_Par5 --h 512 --w 512
#
#python eval.py --method Ours --model_path Pth\\8_8_Aerial_Par6\\Ours\\last.pth \
#       --val_file _Aerial_val.txt --datasetname Aerial_Par6 --h 512 --w 512
#
#python eval.py --method Ours --model_path Pth\\8_8_Aerial_Par7\\Ours\\last.pth \
#       --val_file _Aerial_val.txt --datasetname Aerial_Par7 --h 512 --w 512
#
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
##-----------------------------------------------------------------------#
##   消融实验: 损失函数消融
##-----------------------------------------------------------------------#
## Ours
#python eval.py --method Ours --model_path Pth\\Ubuntu\\Pth\\8_8_Aerial_ablation_onlyseg\\Ours\\last.pth \
#       --val_file _Aerial_val.txt --datasetname Aerial_function_onlyseg --h 512 --w 512
#
#python eval.py --method Ours --model_path Pth\\Ubuntu\\Pth\\8_8_Aerial_ablation_seg_disseg\\Ours\\last.pth \
#       --val_file _Aerial_val.txt --datasetname Aerial_function_seg_disseg --h 512 --w 512
#
#python eval.py --method Ours --model_path Pth\\Ubuntu\\Pth\\8_8_Aerial_ablation_seg_disseg_boundary\\Ours\\last.pth \
#       --val_file _Aerial_val.txt --datasetname Aerial_function_seg_disseg_boundary --h 512 --w 512
#
#python eval.py --method Ours --model_path Pth\\Ubuntu\\Pth\\8_8_Aerial_ablation_seg_disseg_boundary_disboundary\\Ours\\last.pth \
#       --val_file _Aerial_val.txt --datasetname Aerial_function_seg_disseg_boundary_disboundary --h 512 --w 512


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#-----------------------------------------------------------------------#
#   消融实验: 模块消融实验 baseline: bisenetv1
#-----------------------------------------------------------------------#
# resblock
#python eval.py --method ablation --model_path Pth\\Ubuntu\\Pth\\8_9_Aerial_ablation_Module_block\\ablation\\last.pth \
#       --val_file _Aerial_val.txt --datasetname 8_11_Test --h 512 --w 512 --block

#python eval.py --method ablation --model_path Pth\\Ubuntu\\Pth\\8_9_Aerial_ablation_Module_block_flow\\ablation\\last.pth \
#       --val_file _Aerial_val.txt --datasetname Aerial_ablation_Module_block_flow --h 512 --w 512 --block --flow

#python eval.py --method ablation --model_path Pth\\Ubuntu\\Pth\\8_9_Aerial_ablation_Module_block_flow_output\\ablation\\last.pth \
#       --val_file _Aerial_val.txt --datasetname 8_9_Aerial_ablation_Module_block_flow_output --h 128 --w 128 --block --flow --output

#python eval.py --method Ours --model_path Pth\\Ubuntu\\Pth\\8_8_Aerial_loc_loss_3\\Ours\\last.pth \
#       --val_file _Aerial_val.txt --datasetname 8_14_Aerial_3 --h 512 --w 512
#
#python eval.py --method Ours --model_path Pth\\Ubuntu\\Pth\\8_8_Aerial_loc_loss_2\\Ours\\last.pth \
#       --val_file _Aerial_val.txt --datasetname 8_14_Aerial_2 --h 512 --w 512
#
#python eval.py --method Ours --model_path Pth\\Ubuntu\\Pth\\8_8_Aerial_loc_loss_1\\Ours\\last.pth \
#       --val_file _Aerial_val.txt --datasetname 8_14_Aerial_1 --h 512 --w 512
#
#python eval.py --method Ours --model_path Pth\\Ubuntu\\Pth\\7_29_Mas_loc_loss_4\\Ours\\last.pth \
#       --val_file _M_val.txt --datasetname 8_14_Mas_Ours --h 512 --w 512
#
#python eval.py --method Ours --model_path Pth\\Ubuntu\\Pth\\7_29_Mas_loc_loss_1\\Ours\\last.pth \
#       --val_file _M_val.txt --datasetname 8_14_Mas_Ours_1 --h 512 --w 512
#
#python eval.py --method Ours --model_path Pth\\Ubuntu\\Pth\\7_29_Mas_loc_loss_2\\Ours\\last.pth \
#       --val_file _M_val.txt --datasetname 8_14_Mas_Ours_2 --h 512 --w 512
#
#python eval.py --method Ours --model_path Pth\\Ubuntu\\Pth\\7_29_Mas_loc_loss_3\\Ours\\last.pth \
#       --val_file _M_val.txt --datasetname 8_14_Mas_Ours_3 --h 512 --w 512


#python eval.py --method bisenetv1 --model_path Pth\\Ubuntu\\Pth\\Aerial\\bisenetv1\\last.pth\
#       --val_file _Vis2val.txt --datasetname getlabel_A --h 512 --w 512
#
#
#python eval.py --method bisenetv1 --model_path Pth\\Ubuntu\\Pth\\Aerial\\bisenetv1\\last.pth\
#       --val_file _M_val.txt --datasetname getlabel_M --h 512 --w 512

#
#python eval.py --method Ours --model_path Pth\\Ubuntu\\Pth\\8_8_Aerial_loc_loss_2\\Ours\\last.pth \
#       --val_file _Aerial_val.txt --datasetname GT_Aerial --h 512 --w 512
#
#
#
#python eval.py --method Ours --model_path Pth\\Ubuntu\\Pth\\8_8_Aerial_loc_loss_1\\Ours\\last.pth \
#       --val_file _Aerial_val.txt --datasetname 8_15_Aerial_1 --h 512 --w 512

# python3 eval.py --method missformer --model_path /root/data1/Gasking_Segmentation/Pth/Sypase/missformer/last.pth \
#        --val_file test.txt --datasetname Sypase --h 224 --w 224




#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#                                                       Sypase
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# python3 eval.py --method unet --model_path /root/data1/Gasking_Segmentation/Pth/Sypase/unet/last.pth \
#        --val_file test.txt --datasetname Sypase --h 224 --w 224



# python3 eval.py --method resunet --model_path /root/data1/Gasking_Segmentation/Pth/Sypase/resunet/last.pth \
#        --val_file test.txt --datasetname Sypase --h 224 --w 224




# python3 eval.py --method setr --model_path /root/data1/Gasking_Segmentation/Pth/Sypase/setr/last.pth \
#        --val_file test.txt --datasetname Sypase --h 224 --w 224

# python3 eval.py --method missformer --model_path /root/data1/Gasking_Segmentation/Pth/Sypase/missformer/last.pth \
#        --val_file test.txt --datasetname  Sypase --h 224 --w 224


# python3 eval.py --method tsformer --model_path /root/data1/Gasking_Segmentation/Pth/Sypase/tsformer/last.pth \
#        --val_file test.txt --datasetname Sypase --h 224 --w 224


# # --------------------------------------------------------------------------------------------------------------------#
# #                   TODO 还没有跑
# # --------------------------------------------------------------------------------------------------------------------#
# python3 eval.py --method gcn --model_path /root/data1/Gasking_Segmentation/Pth/Sypase/gcn/last.pth \
#        --val_file test.txt --datasetname Sypase --h 224 --w 224

# python3 eval.py --method deeplabv3 --model_path /root/data1/Gasking_Segmentation/Pth/Sypase/deeplabv3/last.pth \
#        --val_file test.txt --datasetname Sypase --h 224 --w 224


# python3 eval.py --method daeformer --model_path /root/data1/Gasking_Segmentation/Pth/Sypase/daeformer/last.pth \
#        --val_file test.txt --datasetname Sypase --h 224 --w 224

# python3 eval.py --method ahfunet --model_path /root/data1/Gasking_Segmentation/Pth/Sypase/ahfunet/last.pth \
#        --val_file test.txt --datasetname Sypase --h 224 --w 224

# python3 eval.py --method Hiformer --model_path /root/data1/Gasking_Segmentation/Pth/Sypase/Hiformer/last.pth \
#        --val_file test.txt --datasetname Sypase --h 224 --w 224


# TODO? 没有跑
# python3 eval.py --method swin_unet --model_path /root/data1/Gasking_Segmentation/Pth/Sypase/swin_unet/last.pth \
#        --val_file test.txt --datasetname Sypase --h 224 --w 224


# python3 eval.py --method transunet --model_path /root/data1/Gasking_Segmentation/Pth/Sypase/transunet/last.pth \
#        --val_file test.txt --datasetname Sypase --h 256 --w 256

python3 eval.py --method fhnet --model_path /root/data1/Gasking_Segmentation_Sypase/Pth/Sypase/fhnet/last.pth \
       --val_file test.txt --datasetname Sypase --h 224 --w 224

