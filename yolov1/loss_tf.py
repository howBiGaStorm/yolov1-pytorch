from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

boundary1 =7*7*20
boundary2 = 7*7*20+7*7*2
cell_size =7
num_class =20
boxes_per_cell =2
image_size = 448
object_scale = 1.0
noobject_scale = 1.0
class_scale = 2.0
coord_scale = 5.0
offset1 = np.transpose(np.reshape(np.array(
            [np.arange(cell_size)] * cell_size * boxes_per_cell),
            (boxes_per_cell, cell_size, cell_size)), (1, 2, 0))


def calc_iou( boxes1, boxes2, scope='iou'):
    """calculate ious
    Args:
      boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
      boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
    Return:
      iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    """

    # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
    boxes1_t = np.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                         boxes1[..., 1] - boxes1[..., 3] / 2.0,
                         boxes1[..., 0] + boxes1[..., 2] / 2.0,
                         boxes1[..., 1] + boxes1[..., 3] / 2.0],
                        axis=-1)

    boxes2_t = np.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                         boxes2[..., 1] - boxes2[..., 3] / 2.0,
                         boxes2[..., 0] + boxes2[..., 2] / 2.0,
                         boxes2[..., 1] + boxes2[..., 3] / 2.0],
                        axis=-1)

    # calculate the left up point & right down point
    lu = np.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
    rd = np.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

    # intersection
    intersection = np.maximum(0.0, rd - lu)
    inter_square = intersection[..., 0] * intersection[..., 1]  # 交集面积

    # calculate the boxs1 square and boxs2 square
    square1 = boxes1[..., 2] * boxes1[..., 3]
    square2 = boxes2[..., 2] * boxes2[..., 3]

    union_square = np.maximum(square1 + square2 - inter_square, 1e-10)  # 并集面积

    return np.clip(inter_square / union_square, 0.0, 1.0)  # 交并比约束在0-1范围内

def loss_layer(predicts_vari, labels,batch_size):  # predict大小为（None,1470），labels大小为floattensor（1,7*7*25）

    predicts = predicts_vari.data.numpy().reshape(batch_size, 1470)
    labels = labels.data
    # 预测结果中的前7*7*20位表示 类别
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

    # 将真实的  labels 转换为相应的矩阵形式
    response = np.reshape(
        labels[..., 0],
        [batch_size, cell_size, cell_size, 1])
    boxes = np.reshape(
        labels[..., 1:5],
        [batch_size, cell_size, cell_size, 1, 4])
    boxes = np.tile(
        boxes, [1, 1, 1, boxes_per_cell, 1]) / image_size
    classes = labels[..., 5:]

    offset = np.reshape(np.array(offset1, dtype=np.float32),[1, cell_size, cell_size, boxes_per_cell])

    offset = np.tile(offset, [batch_size, 1, 1, 1])
    offset_tran = np.transpose(offset, (0, 2, 1, 3))

    # shape为 [4, batch_size, 7, 7, 2]
    predict_boxes_tran = np.stack(
        [(predict_boxes[..., 0] + offset) / cell_size,
         (predict_boxes[..., 1] + offset_tran) / cell_size,
         np.square(predict_boxes[..., 2]),
         np.square(predict_boxes[..., 3])], axis=-1)

    iou_predict_truth = calc_iou(predict_boxes_tran, boxes)

    # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    object_mask = np.max(iou_predict_truth, 3, keepdims=True)
    object_mask = np.array(
        (iou_predict_truth >= object_mask), dtype=np.float32) * response  # 将bool值转化为float值

    # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    noobject_mask = np.ones_like(
        object_mask, dtype=np.float32) - object_mask

    # 参数中加上平方根是对 w 和 h 进行开平方操作
    boxes_tran = np.stack(
        [boxes[..., 0] * cell_size - offset,
         boxes[..., 1] * cell_size - offset_tran,
         np.sqrt(boxes[..., 2]),
         np.sqrt(boxes[..., 3])], axis=-1)

    # class_loss 分类损失
    class_delta = response * (predict_classes - classes)
    class_loss = np.mean(np.sum(np.square(class_delta))) * class_scale

    # object_loss 有目标物体存在的损失
    object_delta = object_mask * (predict_scales - iou_predict_truth)
    object_loss = np.mean(
        np.sum(np.square(object_delta))) * object_scale

    # noobject_loss 没有目标物体时的损失
    noobject_delta = noobject_mask * predict_scales
    noobject_loss = np.mean(
        np.sum(np.square(noobject_delta))) * noobject_scale

    # coord_loss 坐标损失 #shape 为 (batch_size, 7, 7, 2, 1)
    coord_mask = np.expand_dims(object_mask, 4)
    boxes_delta = coord_mask * (predict_boxes - boxes_tran)
    coord_loss = np.mean(np.sum(np.square(boxes_delta))) * coord_scale
    allloss = [class_loss,object_loss, noobject_loss,coord_loss]
    allloss = torch.FloatTensor(allloss)

    return allloss,class_loss,object_loss, noobject_loss,coord_loss