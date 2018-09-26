import matplotlib.pyplot as plt
# import yolov1.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from yolov1.model import *

def calc_iou( boxes1, boxes2, scope='iou'):
    """calculate ious
    Args:
      boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
      boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
    Return:
      iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    """

    # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
    boxes1_t = torch.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                            boxes1[..., 1] - boxes1[..., 3] / 2.0,
                            boxes1[..., 0] + boxes1[..., 2] / 2.0,
                            boxes1[..., 1] + boxes1[..., 3] / 2.0],
                           -1)

    boxes2_t = torch.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                            boxes2[..., 1] - boxes2[..., 3] / 2.0,
                            boxes2[..., 0] + boxes2[..., 2] / 2.0,
                            boxes2[..., 1] + boxes2[..., 3] / 2.0],
                           -1)

    # calculate the left up point & right down point
    lu = boxes1_t[..., :2].max(boxes2_t[..., :2])
    rd = boxes1_t[..., 2:].min(boxes2_t[..., 2:])
    # print(rd-lu)
    # intersection
    zero_ = Variable(torch.zeros((rd - lu).shape)).cuda()
    intersection = zero_.max(rd - lu)
    inter_square = intersection[..., 0] * intersection[..., 1]  # 交集面积

    # calculate the boxs1 square and boxs2 square
    square1 = boxes1[..., 2] * boxes1[..., 3]
    square2 = boxes2[..., 2] * boxes2[..., 3]

    # print(square1 + square2 - inter_square)
    bin_ = torch.ones_like(square1 + square2 - inter_square) * 1e-10
    union_square = bin_.max(square1 + square2 - inter_square)  # 并集面积

    return torch.clamp(inter_square / union_square, 0.0, 1.0)  # 交并比约束在0-1范围内

class myloss(nn.Module):
    def __init__(self,boundary1 =7*7*20,boundary2 = 7*7*20+7*7*2,batch_size = 1,
                 cell_size=7,num_class =20,boxes_per_cell =2,image_size = 448,
                 object_scale=1.0, noobject_scale = 1.0,class_scale = 2.0, coord_scale = 5.0
                 ):
        super(myloss,self).__init__()
        self.boundary1 = boundary1
        self.boundary2 = boundary2
        self.batch_size= batch_size
        self.cell_size = cell_size
        self.num_class = num_class
        self.boxes_per_cell = boxes_per_cell
        self.image_size = image_size
        self.object_scale=object_scale
        self.noobject_scale = noobject_scale
        self.class_scale = class_scale
        self.coord_scale = coord_scale
        self.offsetw = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

    def forward(self,predicts, labels):
        # ------------------------ 预测结果中的前7*7*20位表示 类别---------------------------------
        predicts = predicts.contiguous()
        labels = labels.contiguous()
        predict_classes = predicts[:, :self.boundary1].contiguous().view([self.batch_size, self.cell_size, self.cell_size,self.num_class])
        # 预测结果中7*7*20之后的7*7*2 表示 每个网格中两个框中有物体的置信度
        predict_scales = predicts[:, self.boundary1:self.boundary2].contiguous().view([self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
        # 最后7*7*2*4位表示 每个网格中的2个框的4个位置信息
        predict_boxes = predicts[:, self.boundary2:].contiguous().view([self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

        # ----------------------将真实的  labels 转换为相应的矩阵形式-------------------------
        response = labels[..., 0].contiguous().view([self.batch_size, self.cell_size, self.cell_size, 1])
        boxes = labels[..., 1:5].contiguous().view([self.batch_size, self.cell_size, self.cell_size, 1, 4])
        # print(type(boxes))
        boxes = boxes.repeat(1, 1, 1, self.boxes_per_cell, 1) / self.image_size
        # print(type(boxes))
        classes = labels[..., 5:]
        # print(type(classes))
        # # ---------------------------------offset-------------------------------------------------
        offset1 = Variable(torch.FloatTensor(self.offsetw)).cuda()
        offset = offset1.contiguous().view([1, self.cell_size, self.cell_size, self.boxes_per_cell])
        # print(offset.shape)
        offset = offset.repeat(self.batch_size, 1, 1, 1)
        offset_tran = offset.permute(0, 2, 1, 3)
        # print(type(offset_tran))
        #
        # # ------------------------------------------------------------------------------------------
        # shape为 [4, batch_size, 7, 7, 2]
        predict_boxes_tran = torch.stack(
            [(predict_boxes[..., 0] + offset) / self.cell_size,
             (predict_boxes[..., 1] + offset_tran) / self.cell_size,
             torch.pow(predict_boxes[..., 2], 2),
             torch.pow(predict_boxes[..., 3], 2)], -1)
        # print(type(predict_boxes_tran))
        #
        iou_predict_truth = calc_iou(predict_boxes_tran, boxes).cuda()
        # print(iou_predict_truth)
        # # -------------------------------------------------------------------------------------------
        # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        object_mask, _ = torch.max(iou_predict_truth, 3, True)

        # print(type(object_mask))

        # object_mask = torch.ByteTensor(object_mask)
        # object_mask = np.array(
        #     (iou_predict_truth >= object_mask), dtype=np.float32) * response  # 将bool值转化为float值
        object_mask = torch.ge(iou_predict_truth, object_mask).float()
        object_mask = torch.mul(object_mask, response)
        # print(object_mask)
        # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        noobject_mask = torch.ones_like(object_mask) - object_mask
        # print(noobject_mask)
        # # 参数中加上平方根是对 w 和 h 进行开平方操作
        boxes_tran = torch.stack(
            [boxes[..., 0] * self.cell_size - offset,
             boxes[..., 1] * self.cell_size - offset_tran,
             torch.sqrt(boxes[..., 2]),
             torch.sqrt(boxes[..., 3])], -1)
        # print(boxes_tran)
        #
        # class_loss 分类损失
        class_delta = response * (predict_classes - classes)
        class_loss = torch.mean(torch.pow(class_delta, 2).sum(3).sum(2).sum(1)) * self.class_scale
        # print(class_loss)

        # object_loss 有目标物体存在的损失
        object_delta = object_mask * (predict_scales - iou_predict_truth)
        object_loss = torch.mean(torch.pow(object_delta, 2).sum(3).sum(2).sum(1)) * self.object_scale

        # noobject_loss 没有目标物体时的损失
        noobject_delta = noobject_mask * predict_scales
        noobject_loss = torch.mean(torch.pow(noobject_delta, 2).sum(3).sum(2).sum(1)) * self.noobject_scale

        # coord_loss 坐标损失 #shape 为 (batch_size, 7, 7, 2, 1)
        coord_mask = torch.unsqueeze(object_mask, 4)
        boxes_delta = coord_mask * (predict_boxes - boxes_tran)
        coord_loss = torch.mean(torch.pow(boxes_delta, 2).sum(4).sum(3).sum(2).sum(1)) * self.coord_scale

        allloss = class_loss + object_loss + noobject_loss + coord_loss

        return allloss, class_loss, object_loss, noobject_loss, coord_loss