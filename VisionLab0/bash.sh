##+++++++++++++++++++++++++++++++++++++++++++++#
##            kvasir_seg 12.5k
##+++++++++++++++++++++++++++++++++++++++++++++#
#python train.py  --save 100 --freeze_epoch 250 --unfreeze_epoch 1000  --datasetname kvasir_seg \
#--bs 64  --train_txt kvasir_seg_train.txt --test_txt kvasir_seg_val.txt \
#     --val_txt kvasir_seg_val.txt --method unet
#
##-----------------------------------------------#
## unetplus
##-----------------------------------------------#
#python train.py  --save 100 --freeze_epoch 250 --unfreeze_epoch 1000   --datasetname kvasir_seg \
#    --bs 64  --train_txt kvasir_seg_train.txt --test_txt kvasir_seg_val.txt \
#     --val_txt kvasir_seg_val.txt --method unetplus
#
## resunet
#python train.py  --save 100 --freeze_epoch 250 --unfreeze_epoch 1000   --datasetname kvasir_seg \
#    --bs 64  --train_txt kvasir_seg_train.txt --test_txt kvasir_seg_val.txt \
#     --val_txt kvasir_seg_val.txt --method resunet
#
## resunteplus
#python train.py  --save 100 --freeze_epoch 500 --unfreeze_epoch 2000   --datasetname kvasir_seg \
#    --bs 32  --train_txt kvasir_seg_train.txt --test_txt kvasir_seg_val.txt \
#     --val_txt kvasir_seg_val.txt --method resunetplus
#
## setr
#python train.py  --save 100 --freeze_epoch 500 --unfreeze_epoch 2000  --datasetname kvasir_seg \
#    --bs 32  --train_txt kvasir_seg_train.txt --test_txt kvasir_seg_val.txt \
#     --val_txt kvasir_seg_val.txt --method setr
#
## transunet
#python train.py  --save 100 --freeze_epoch 500 --unfreeze_epoch 2000  --datasetname kvasir_seg \
#    --bs 32  --train_txt kvasir_seg_train.txt --test_txt kvasir_seg_val.txt \
#     --val_txt kvasir_seg_val.txt --method transunet
#
#

## swin_unet
#python train.py  --save 100 --freeze_epoch 500 --unfreeze_epoch 2000  --datasetname kvasir_seg \
#    --bs 32  --train_txt kvasir_seg_train.txt --test_txt kvasir_seg_val.txt \
#     --val_txt kvasir_seg_val.txt --method swin_unet
#
## mmisformer
python train.py  --save 100 --freeze_epoch 500 --unfreeze_epoch 2000  --datasetname kvasir_seg \
    --bs 32  --train_txt kvasir_seg_train.txt --test_txt kvasir_seg_val.txt \
     --val_txt kvasir_seg_val.txt --method mmisformer





#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# kvasir_instrument
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#python train.py  --save 100 --freeze_epoch 500 --unfreeze_epoch 2000  --datasetname kvasir_instrument \
#--bs 64  --train_txt kvasir_instrument_train.txt --test_txt kvasir_instrument_test.txt \
#     --val_txt kvasir_seg_val.txt --method unet
#
##-----------------------------------------------#
## unetplus
##-----------------------------------------------#
#python train.py  --save 100 --freeze_epoch 500 --unfreeze_epoch 2000   --datasetname kvasir_instrument \
#--bs 64  --train_txt kvasir_instrument_train.txt --test_txt kvasir_instrument_test.txt \
#     --val_txt kvasir_instrument_test.txt --method unetplus

# resunet   kvasir_instrument_train.txt
#python train.py  --save 100 --init_epoch 800 --freeze_epoch 500 --unfreeze_epoch 2000   --datasetname kvasir_instrument \
#    --bs 64  --train_txt kvasir_instrument_train.txt --test_txt kvasir_instrument_test.txt \
#     --val_txt kvasir_instrument_test.txt --method resunet  --model_path Pth/kvasir_instrument/resunet/800.pth
#
## resunteplus
#python train.py  --save 100 --freeze_epoch 800 --unfreeze_epoch 3500   --datasetname kvasir_instrument \
#    --bs 32  --train_txt kvasir_instrument_train.txt --test_txt kvasir_instrument_test.txt \
#     --val_txt kvasir_instrument_test.txt --method resunetplus
#
## setr
#python train.py  --save 100 --freeze_epoch 800 --unfreeze_epoch 3500  --datasetname kvasir_instrument \
#    --bs 32  --train_txt kvasir_instrument_train.txt --test_txt kvasir_instrument_test.txt \
#     --val_txt kvasir_instrument_test.txt --method setr

# transunet
#python train.py  --save 100 --freeze_epoch 800 --unfreeze_epoch 3500  --datasetname kvasir_instrument \
#    --bs 32  --train_txt kvasir_instrument_train.txt --test_txt kvasir_instrument_test.txt \
#     --val_txt kvasir_instrument_test.txt --method transunet


# swin_unet
#python train.py  --save 100 --freeze_epoch 800 --unfreeze_epoch 3500  --datasetname kvasir_instrument \
#    --bs 32  --train_txt kvasir_instrument_train.txt --test_txt kvasir_instrument_test.txt \
#     --val_txt kvasir_instrument_test.txt --method swin_unet

# mmisformer
#python train.py  --save 100 --freeze_epoch 800 --unfreeze_epoch 3500  --datasetname kvasir_instrument \
#     --bs 32  --train_txt kvasir_instrument_train.txt --test_txt kvasir_instrument_test.txt \
#      --val_txt kvasir_instrument_test.txt --method missformer



##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
## kvasir_Capsule
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#python train.py  --save 100 --freeze_epoch 1000 --unfreeze_epoch 5000  --datasetname kvasir_Capsule \
#--bs 64  --train_txt kvasir_Capsule_train.txt --test_txt kvasir_Capsule_test.txt \
#     --val_txt kvasir_Capsule_test.txt --method unet
#
##-----------------------------------------------#
## unetplus
##-----------------------------------------------#
#python train.py  --save 100 --freeze_epoch 1000 --unfreeze_epoch 5000  --datasetname kvasir_Capsule \
#--bs 64  --train_txt kvasir_Capsule_train.txt --test_txt kvasir_Capsule_test.txt \
#     --val_txt kvasir_Capsule_test.txt --method unetplus
#
## resunet
#python train.py  --save 100 --freeze_epoch 1000 --unfreeze_epoch 5000  --datasetname kvasir_Capsule \
#--bs 64  --train_txt kvasir_Capsule_train.txt --test_txt kvasir_Capsule_test.txt \
#     --val_txt kvasir_Capsule_test.txt --method resunet
#
## resunteplus
#python train.py  --save 100 --freeze_epoch 1000 --unfreeze_epoch 5000  --datasetname kvasir_Capsule \
#--bs 64  --train_txt kvasir_Capsule_train.txt --test_txt kvasir_Capsule_test.txt \
#     --val_txt kvasir_Capsule_test.txt --method resunetplus
#
## setr
#python train.py --save 100 --freeze_epoch 1000 --unfreeze_epoch 5000  --datasetname kvasir_Capsule \
#--bs 64  --train_txt kvasir_Capsule_train.txt --test_txt kvasir_Capsule_test.txt \
#     --val_txt kvasir_Capsule_test.txt --method setr
#
## transunet
#python train.py  --save 100 --freeze_epoch 1000 --unfreeze_epoch 5000  --datasetname kvasir_Capsule \
#--bs 64  --train_txt kvasir_Capsule_train.txt --test_txt kvasir_Capsule_test.txt \
#     --val_txt kvasir_Capsule_test.txt --method transunet
#
#
## swin_unet
#python train.py  --save 100 --freeze_epoch 1000 --unfreeze_epoch 5000  --datasetname kvasir_Capsule \
#--bs 64  --train_txt kvasir_Capsule_train.txt --test_txt kvasir_Capsule_test.txt \
#     --val_txt kvasir_Capsule_test.txt --method swin_unet
#
## mmisformer
#python train.py  --save 100 --freeze_epoch 1000 --unfreeze_epoch 5000  --datasetname kvasir_Capsule \
#--bs 64  --train_txt kvasir_Capsule_train.txt --test_txt kvasir_Capsule_test.txt \
#     --val_txt kvasir_Capsule_test.txt --method missformer



#python train.py  --save 10 --freeze_epoch 800 --unfreeze_epoch 3500  --datasetname kvasir_instrument \
#     --bs 32  --train_txt kvasir_instrument_train.txt --test_txt kvasir_instrument_test.txt \
#      --val_txt kvasir_instrument_test.txt --method tsformer --init_epoch 1100 --model_path Pth\\1100.pth


# python train.py  --save 10 --freeze_epoch 500 --unfreeze_epoch 2000  --datasetname kvasir_seg_module \
#    --bs 20  --train_txt kvasir_seg_train.txt --test_txt kvasir_seg_val.txt \
#     --val_txt kvasir_seg_val.txt --method tsformer --init_epoch 760 --model_path Pth/kvasir_seg_module/tsformer/760.pth




# 没有预训练的transunet

## transunet
# python3 train.py  --save 20 --freeze_epoch 500 --init_epoch 80 --unfreeze_epoch 2000  --datasetname kvasir_seg_nopretrained \
#     --bs 2 --train_txt kvasir_seg_train.txt --test_txt kvasir_seg_val.txt --num_worker 4 \
#      --val_txt kvasir_seg_val.txt --method ftmnet #--model_path /root/data1/Gasking_Segmentation/Pth/kvasir_seg_nopretrained/ftmnet/80.pth


# python3 train.py  --save 100 --freeze_epoch 500 --init_epoch 3200 --unfreeze_epoch 3500  --datasetname kvasir_seg_nopretrained \
#     --bs 32 --train_txt kvasir_seg_train.txt --test_txt kvasir_seg_val.txt --num_worker 4\
#      --val_txt kvasir_seg_val.txt --method funet --model_path /root/data1/Gasking_Segmentation/Pth/kvasir_seg_nopretrained/funet/3200.pth


# python3 train.py  --save 100 --freeze_epoch 500 --init_epoch 0 --unfreeze_epoch 2000  --datasetname kvasir_seg_nopretrained \
#     --bs 32 --train_txt kvasir_seg_train.txt --test_txt kvasir_seg_val.txt --num_worker 4\
#      --val_txt kvasir_seg_val.txt --method protounet


# python3 train.py  --save 10 --freeze_epoch 500 --init_epoch 1804 --unfreeze_epoch 3500  --datasetname kvasir_seg \
#       --bs 24 --train_txt kvasir_seg_train.txt --test_txt kvasir_seg_val.txt --num_worker 4 \
#      --val_txt kvasir_seg_val.txt --method fhnet --model_path /root/data1/Gasking_Segmentation/Pth/kvasir_seg/fhnet/best.pth


#---------------------------------------------------------------------------------#
#                 Sypase 数据集
#---------------------------------------------------------------------------------#
# python3 train.py  --save 10 --freeze_epoch 250 --init_epoch 510 --unfreeze_epoch 800  --datasetname Sypase \
#       --bs 32 --train_txt train.txt --test_txt test.txt --num_worker 4 \
#      --val_txt test.txt --method unet --model_path /root/data1/Gasking_Segmentation/Pth/Sypase/unet/best.pth


# python3 train.py  --save 10 --freeze_epoch 250 --init_epoch 0 --unfreeze_epoch 800  --datasetname Sypase \
#       --bs 32 --train_txt train.txt --test_txt test.txt --num_worker 4 \
#      --val_txt test.txt --method unetplus 


# python3 train.py  --save 10 --freeze_epoch 250 --init_epoch 0 --unfreeze_epoch 800  --datasetname Sypase \
#       --bs 32 --train_txt train.txt --test_txt test.txt --num_worker 4 \
#      --val_txt test.txt --method missformer 

# python3 train.py  --save 10 --freeze_epoch 250 --init_epoch 0 --unfreeze_epoch 800  --datasetname Sypase \
#       --bs 32 --train_txt train.txt --test_txt test.txt --num_worker 4 \
#      --val_txt test.txt --method resunet 

# python3 train.py  --save 10 --freeze_epoch 250 --init_epoch 0 --unfreeze_epoch 800  --datasetname Sypase \
#       --bs 32 --train_txt train.txt --test_txt test.txt --num_worker 4 \
#      --val_txt test.txt --method resunetplus 


# python3 train.py  --save 10 --freeze_epoch 250 --init_epoch 0 --unfreeze_epoch 800  --datasetname Sypase \
#       --bs 32 --train_txt train.txt --test_txt .txt --num_worker 4\
#       --val_txt test.txt --method transunet


python3 train.py  --save 10 --freeze_epoch 250 --init_epoch 0 --unfreeze_epoch 800  --datasetname Sypase \
      --bs 32 --train_txt train.txt --test_txt test.txt --num_worker 4\
      --val_txt test.txt --method swin_unet


# TODO 没有色彩层面
python3 train.py  --save 10 --freeze_epoch 250 --init_epoch 0 --unfreeze_epoch 800  --datasetname Sypase \
      --bs 32 --train_txt train.txt --test_txt test.txt --num_worker 4\
      --val_txt test.txt --method setr

# python3 train.py  --save 10 --freeze_epoch 250 --init_epoch 0 --unfreeze_epoch 800  --datasetname Sypase \
#       --bs 32 --train_txt train.txt --test_txt test.txt --num_worker 4\
#       --val_txt test.txt --method tsformer


# python3 train.py  --save 50 --freeze_epoch 250 --init_epoch 800 --unfreeze_epoch 1000  --datasetname Sypase \
#       --bs 32 --train_txt train.txt --test_txt test.txt --num_worker 4\
#       --val_txt test.txt --method gcn --model_path /root/data1/Gasking_Segmentation/Pth/Sypase/gcn/last.pth



# python3 train.py  --save 100 --freeze_epoch 250 --init_epoch 0 --unfreeze_epoch 800  --datasetname Sypase \
#       --bs 32 --train_txt train.txt --test_txt test.txt --num_worker 4\
#       --val_txt test.txt --method deeplabv3

# python3 train.py  --save 100 --freeze_epoch 250 --init_epoch 0 --unfreeze_epoch 800  --datasetname Sypase \
#       --bs 32 --train_txt train.txt --test_txt test.txt --num_worker 4\
#       --val_txt test.txt --method pspnet




# TODO? 权重文件
# python3 train.py  --save 100 --freeze_epoch 250 --init_epoch 0 --unfreeze_epoch 800  --datasetname Sypase \
#       --bs 32 --train_txt train.txt --test_txt test.txt --num_worker 4\
#       --val_txt test.txt --method Hiformer

# python3 train.py  --save 100 --freeze_epoch 250 --init_epoch 0 --unfreeze_epoch 800  --datasetname Sypase \
#       --bs 32 --train_txt train.txt --test_txt test.txt --num_worker 4\
#       --val_txt test.txt --method ahfunet

# #daeformer
# python3 train.py  --save 100 --freeze_epoch 250 --init_epoch 0 --unfreeze_epoch 800  --datasetname Sypase \
#       --bs 32 --train_txt train.txt --test_txt test.txt --num_worker 4\
      # --val_txt test.txt --method daeformer

# python3 train.py  --save 100 --freeze_epoch 250 --init_epoch 0 --unfreeze_epoch 800  --datasetname Sypase \
#       --bs 32 --train_txt train.txt --test_txt test.txt --num_worker 4\
#       --val_txt test.txt --method dconnNet
