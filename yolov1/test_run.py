
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import shutil  # 高级的文件操作模块
from collections import OrderedDict  # 使字典的Key是有序的
import matplotlib.pyplot as plt


import torch.optim as optim
from torch.utils.data import DataLoader

import data
from yolov1.loss_layer import *
from yolov1.loss_tf import *
from yolov1.model import YOLO
from data.data_tf import *
import torchvision.models as models  # 子模块为一些构造好的模型
import argparse  # 命令行调用
from yolov1 import train
import torch.nn.init as init
from yolov1.detector import *

print('loading model...')
vgg = models.vgg16(False)  # 采用vgg16预训练好的模型
# vgg.load_state_dict(torch.load('vgg16-397923af.pth'))
model = YOLO(vgg.features)
state_dict = torch.load('my_model_epoch_49.pkl')
model.load_state_dict(state_dict)
model.cuda()
print('model load complete!')
criteria = myloss(batch_size=1).cuda()

for i in range(20,30):

    if i<10:
        pic_num = '00000'+str(i)
    elif i<100:
        pic_num = '0000' + str(i)
    elif i <1000:
        pic_num = '000' + str(i)
    elif i<10000:
        pic_num = '00' + str(i)

    img_org = cv2.imread('/home/neec10601/桌面/yolo-pytorch-master/data/VOCdevkit_small/VOC2007t/JPEGImages/'+pic_num+'.jpg')
    img_h, img_w, _ = img_org.shape  # 图片大小
    image_size = 448
    inputs = cv2.resize(img_org, (image_size, image_size))  # 图像缩放
    # inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)  # 调整图片频道，BGR -> RGB
    # inputs = (inputs / 255.0) * 2.0 - 1.0  # 调整像素值 在-1,1之间分布
    # inputs = np.transpose(np.array(inputs, dtype=np.float32), (2, 0, 1))
    inputs = np.reshape(inputs, (1, image_size, image_size,3))
    inputs = np.transpose(inputs, (0, 3, 1, 2))

    img1 = Variable(torch.FloatTensor(inputs)).cuda()

    # dataset = VOCdetection(root=VOC_ROOT)
    # dataloader = data.DataLoader(dataset, 1, shuffle=False)
    # for (img, gt, height, width) in dataloader:
    #     img2 = img
    #     # print('img2.shape:',img2.shape)
    #     labels = Variable(gt).cuda()
    # img2 = Variable(img2).cuda()
    # print(img1.data == img2.data)
    # output2 = model(img2)
    # loss2,a2,b2,c2,d2 = criteria(output2,labels)

    output1 = model(img1)


    # loss1,a1,b1,c1,d1 = criteria(output1,labels)
    # print('test_pic_loss:',loss1.data)
    # print('train_pic_loss:',loss2.data)

    output = output1.data.cpu().numpy().reshape(1470)
    detect = Detector()
    result = detect.interpret_output(output)
    for i in range(len(result)):
        result[i][1] *= (1.0 * img_w / image_size)
        result[i][2] *= (1.0 * img_h / image_size)
        result[i][3] *= (1.0 * img_w / image_size)
        result[i][4] *= (1.0 * img_h / image_size)
    detect.draw_result(img_org,result)

    cv2.imshow('Image_' + pic_num, img_org)
    k = cv2.waitKey(0)
    if k == ord('s'):
        cv2.imwrite('./result_pic/Image_'+pic_num+'.jpg',img_org)
        # cv2.destroyAllWindows()
    else:
        # cv2.destroyAllWindows()
        continue
cv2.destroyAllWindows()
