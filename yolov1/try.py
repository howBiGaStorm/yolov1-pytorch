
import matplotlib.pyplot as plt
# import yolov1.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from yolov1.model import *

import sys
sys.path.append('/home/neec10601/桌面/yolo-pytorch-master/data')
from data.data_tf import *
from yolov1.loss_tf import *
from yolov1.model import *
import torchvision.models as models
dataset = VOCdetection(root = VOC_ROOT)
batch_size = 2
dataloader = data.DataLoader(dataset=dataset,batch_size = batch_size,shuffle=False)
dataload = iter(dataloader)
target_list = []
w1 = []
h2=[]
batch_num = 0
for (img,gt,width,height) in dataloader:
    img = Variable(img.cuda())
    # for key in gt.keys():
    #     gt[key] = Variable(gt[key])
    # target_list.append(gt)
    w1.append(width)
    h2.append(height)
    batch_num += 1
print(img)
print(gt[...,1:5])

# img = Variable(img)
# vgg = models.vgg16(False)  # 采用vgg16预训练好的模型
# vgg.load_state_dict(torch.load('vgg16-397923af.pth'))
# model = YOLO(vgg.features)
#
# y_pred = model(img)
# # print(y_pred.data[0])
#
# # predicts_torch = torch.randn(1,1470)
# # labels = torch.randn(1,7,7,25)
# # print(labels.shape)
# predicts_torch = y_pred
# labels = gt
#
#
# labels = labels.numpy()
# print(predicts_torch.shape)
# print(labels.shape)

'''
boundary1 =7*7*20
boundary2 = 7*7*20+7*7*2
batch_size =2
cell_size =7
num_class =20
boxes_per_cell =2
image_size = 448
offset1 = np.transpose(np.reshape(np.array(
            [np.arange(cell_size)] * cell_size * boxes_per_cell),
            (boxes_per_cell, cell_size, cell_size)), (1, 2, 0))
'''
#
# allloss,a,b,c,d = loss_layer(predicts_torch,labels,batch_size)
# sum1 = np.array([a,b,c,d])
# loss = [np.sum(sum1)]
# l = torch.FloatTensor(loss)
# print(type(torch.FloatTensor(loss)))
# # loss = torch.sum(allloss)
# # print(allloss)
# # print(loss)

# print(type(loss))
'''
# -----------------------------predict----------------------------------
predict_classes = np.reshape(
        predicts[:, :boundary1],
        [batch_size, cell_size, cell_size, num_class])
    # 预测结果中7*7*20之后的7*7*2 表示 每个网格中两个框中有物体的置信度
predict_scales = np.reshape(
    predicts[:, boundary1:boundary2],
    [batch_size, cell_size, cell_size, boxes_per_cell])
# 最后7*7*2*4位表示 每个网格中的2个框的4个位置信息
predict_boxes = np.reshape(
    predicts[:, boundary2:],
    [batch_size, cell_size, cell_size, boxes_per_cell, 4])
print('predict_boxes:',predict_boxes.shape)
# ------------------------GTchuli--------------------------------------
response = np.reshape(
    labels[..., 0],
    [batch_size, cell_size, cell_size, 1])
boxes = np.reshape(
    labels[..., 1:5],
    [batch_size, cell_size, cell_size, 1, 4])
boxes = np.tile(
    boxes, [1, 1, 1, boxes_per_cell, 1]) / image_size
classes = labels[..., 5:]
print('boxes:',boxes.shape)
# -------------------------------offset---------------------------------------
offset = np.reshape(np.array(offset1, dtype=np.float32),[1, cell_size, cell_size, boxes_per_cell])
offset = np.tile(offset, [batch_size, 1, 1, 1])
offset_tran = np.transpose(offset, (0, 2, 1, 3))
print('offset_trans:',offset_tran.shape)
# --------------------------trans--------------------------------------------
predict_boxes_tran = np.stack(
        [(predict_boxes[..., 0] + offset) / cell_size,
         (predict_boxes[..., 1] + offset_tran) / cell_size,
         np.square(predict_boxes[..., 2]),
         np.square(predict_boxes[..., 3])], axis=-1)
print('predict_boxes_trans:',predict_boxes_tran.shape)
# --------------------------cal_iou--------------------------------------
iou_predict_truth = calc_iou(predict_boxes_tran, boxes)
print('iou:',iou_predict_truth.shape)
# -------------------------cal object mask-----------------------------
object_mask = np.max(iou_predict_truth, 3,keepdims=True)
object_mask = np.array(
    (iou_predict_truth >= object_mask), dtype=np.float32) * response
print('object_mask:',object_mask.shape)

noobject_mask = np.ones_like(
        object_mask, dtype=np.float32) - object_mask
print('noobject_mask:',noobject_mask.shape)
# ---------------------------boxes trans---------------------------
boxes_tran = np.stack(
        [boxes[..., 0] * cell_size - offset,
         boxes[..., 1] * cell_size - offset_tran,
         np.sqrt(boxes[..., 2]),
         np.sqrt(boxes[..., 3])], axis=-1)
print('boxes_tran:',boxes_tran.shape)
# -------------------------------class loss ------------------------------
class_delta = response * (predict_classes - classes)
class_loss = np.mean(np.sum( np.square(class_delta))) * class_scale
print('class loss：',class_loss)

object_delta = object_mask * (predict_scales - iou_predict_truth)
object_loss = np.mean(np.sum(np.square(object_delta))) * object_scale
print('object loss:',object_loss)

noobject_delta = noobject_mask * predict_scales
noobject_loss = np.mean(np.sum(np.square(noobject_delta))) * noobject_scale
print('no object loss:',noobject_loss)

coord_mask = np.expand_dims(object_mask, 4)
boxes_delta = coord_mask * (predict_boxes - boxes_tran)
coord_loss = np.mean(np.sum(np.square(boxes_delta))) * coord_scale
print('coord loss:',coord_loss)

'''









