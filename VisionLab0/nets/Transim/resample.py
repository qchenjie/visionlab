import torch
import cv2
from PIL import Image
import numpy as np
from nets.Transim.sample import StyleTransfer
from nets.Transim.sample import ssim, pnsr
from nets.Unet.Unet import UNet
import torch.nn.functional as F

def predict(path = None):
    assert path != None, "__PATH__ ERROR!"
    model_path = 'model_last_0.pth'

    input_shape = (256, 256)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = StyleTransfer(fs = 256)

    model.load_state_dict(torch.load(model_path, map_location = device))
    model = model.eval()
    model = model.to(device)

    image = Image.open(path).resize(input_shape)

    x1 = np.array(image)



    cv2.imshow('im0',x1[:,:,::-1])

    image = np.array(image) / 255.

    image = image.transpose((2, 0, 1))

    x = torch.from_numpy(image[ None, :, :, : ]).float().to(device)

    out = model(x)

    # boundary = torch.sigmoid(boundary[0]).detach().cpu().numpy().transpose((1,2,0))
    #
    # point = F.interpolate(point,size = boundary.shape[:2],mode = 'bilinear',align_corners = True)
    # point = torch.sigmoid(point[0]).detach().cpu().numpy().transpose((1,2,0)) > 0.64
    # boundary = boundary *255
    # boundary = boundary.astype(np.uint8)
    #
    # point = point * 255
    # point = point.astype(np.uint8)
    # print(point.shape)
    # print(mean.max(),std.max())

    # out = torch.nn.Sigmoid()(out)

    # print(out.max())
    output = F.sigmoid(out)
    output = output.cpu().detach().numpy()[ 0 ] * 255.

    output = output.astype(np.uint8)

    _x1 = cv2.cvtColor(x1, cv2.COLOR_RGB2LAB)

    l, a, b = _x1[ :, :, 0 ], _x1[ :, :, 1 ], _x1[ :, :, 2 ]
    pl, pa, pb = output[ :, 0 ], output[ :, 1 ], output[ :, 2 ]
    print(pl.shape, l.shape)

    newl = pl[ l ]
    newa = pa[ a ]
    newb = pb[ b ]

    im = cv2.merge([ newl, newa, newb ])
    im = cv2.cvtColor(im, cv2.COLOR_LAB2BGR)

    print(np.max(im),np.min(im))

    print(ssim(im, x1), pnsr(im, x1))

    cv2.imshow('im', im)
    # cv2.imshow('boundary',boundary)
    # cv2.imshow('points',point)

    cv2.waitKey(0)


if __name__ == '__main__':

    path = '/root/data1/Gasking_Segmentation/datasets/kvasir-seg/images/cju2r6mt2om21099352pny5gw.jpg'

    predict(path)