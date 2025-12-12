from PIL import Image
import cv2
import numpy as np


def gaussian_radius(det_size, min_overlap=0.7):
    box_w, box_h  = det_size
    a1 = 1
    b1 = (box_w + box_h)
    c1 = box_w * box_h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (box_w + box_h)
    c2 = (1 - min_overlap) * box_w * box_h
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (box_w + box_h)
    c3 = (min_overlap - 1) * box_w * box_h
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)

def draw_gassuian(heat,x,y,radius,feature_w,feature_h):

    sigma = radius

    for j in range(y-3 * int(radius) - 1,y+3 * int(radius) + 1):
     for i in range(x-3 * int(radius) - 1,x+3 * int(radius) + 1):
      if (i < feature_w) and j < (feature_h):
         heat[j,i] = np.exp(- (i - x)**2 / (2*sigma**2) - (j - y)**2 / (2*sigma**2))

         print(heat_map[j,i])
    return heat


path = '1011.tif'

image = Image.open(path).convert('RGB')
image = np.array(image)

h,w,_ = np.array(image).shape
stride = 2

heat_map = np.zeros((h//stride,w//stride))

image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

heri ,_ = cv2.threshold(image,0,200,cv2.THRESH_BINARY)

ksize = np.array([-1,1,1,-1,8,-1,-1,1,1]).reshape((-1,3))
lap = cv2.Laplacian(_,cv2.CV_8U,ksize)
#----------------------------------------#
#             腐蚀运算
#----------------------------------------#

# 使用拉普拉斯变换得到轮廓效果好一些
# image = cv2.erode(_,kernel = (3,3),iterations = 3)
lap = cv2.dilate(lap,kernel = (2,2))

canny = cv2.Canny(image,0,180)


cv2.imshow('canny',canny)

boundary = np.zeros_like(canny)

boundary[canny > 0.1] = 1


judge,count = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# 坐标点
boxes = []


for box in judge:
    boxes.extend([cv2.boundingRect(np.array(box))])

boxes = np.array(boxes)

feature_h,feature_w = heat_map.shape


for box in boxes:
    x1,y1 = box[0],box[1]
    x2,y2 = x1 + box[2],y1 + box[3]
    cv2.rectangle(image,(x1,y1),(x2,y2),
                  (255,0,0),2,cv2.LINE_AA)
    center_x = (x1 + x2) // 2 // stride
    center_y = (y1 + y2) // 2 // stride
    w = box[2] // stride
    h = box[3] // stride

    print(f'w:{w},h:{h}')
    # TODO: reference
    radius =  0.5 * gaussian_radius((w, h))  / 3
    draw_gassuian(heat_map,center_x,center_y,radius,feature_w,feature_h)

    #heat_map[center_y,center_x] = 255



cv2.imshow('im0', image)
cv2.imshow('heat', heat_map *255)
cv2.imshow('boundary',boundary * 255)
cv2.imshow('lap',lap)
cv2.waitKey(0)



