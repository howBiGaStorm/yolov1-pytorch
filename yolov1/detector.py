import argparse
import cv2
import torch
import os
from torch.autograd import Variable
import numpy as np
from yolov1.model import *
from collections import OrderedDict  # 使字典的Key是有序的
import torchvision.models as models  # 子模块为一些构造好的模型
import sys
sys.path.append('E:\grade_2_up\computer-Vision\yolo-pytorch-master\data')
from data.data_tf import *

class Detector(object):

    def __init__(self):

        self.classes = VOC_CLASSES  # 类别
        self.num_class = len(self.classes)  # 类别数
        self.image_size = 448  # 图片大小 448*448*3
        self.cell_size = 7
        self.boxes_per_cell = 2
        self.threshold = 0.2
        self.iou_threshold = 0.5
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 +\
            self.cell_size * self.cell_size * self.boxes_per_cell

    def interpret_output(self, output):
        # 概率零矩阵，7*7*2*20
        probs = np.zeros((self.cell_size, self.cell_size,
                          self.boxes_per_cell, self.num_class))
        # print('第一个格子的类别序列',output[0:20])
        class_probs = np.reshape(
            output[0:self.boundary1],
            (self.cell_size, self.cell_size, self.num_class))
        # print('前两个格子的啥',output[self.boundary1:self.boundary1+4])
        scales = np.reshape(
            output[self.boundary1:self.boundary2],
            (self.cell_size, self.cell_size, self.boxes_per_cell))
        # print('第一个格子的框框',output[self.boundary2:self.boundary2+8])
        boxes = np.reshape(
            output[self.boundary2:],
            (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
        # offset的大小:(14,7)
        offset = np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell)
        # print('offset',offset)
        # offset的大小:(7,7,2)
        offset = np.transpose(
            np.reshape(
                offset,
                [self.boxes_per_cell, self.cell_size, self.cell_size]),
            (1, 2, 0))
        # print('offset', offset)

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        boxes *= self.image_size
        # print(boxes.shape)
        # 置信度为类别概率乘以框框中有目标的概率
        for i in range(self.boxes_per_cell):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i])
        # 7*7*2个框，框里的是20个类的概率，超过0.2的留下,T OR f
        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        # print('filter_mat_probs',filter_mat_probs)
        # print('filter_mat_probs.shape',filter_mat_probs.shape)

        # 不为0的元素的下标，即这些留下概率的下标
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        # print('filter_mat_boxes',filter_mat_boxes)
        # 满足概率阈值的框框们的四个位置信息
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]
        # print('boxes:',boxes_filtered)
        # 满足概率阈值的具体概率值，有几个框框，就有几个概率值
        probs_filtered = probs[filter_mat_probs]

        # 满足概率阈值的概率中最大的值的下标
        classes_num_filtered = np.argmax(
            filter_mat_probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
        # print('classes_num_filtered',classes_num_filtered)

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]
        # print('classes:',classes_num_filtered)

        # 把可能重复框定的，即框框重叠的，IOU大的抹去
        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0
        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]
        # print('classes_num_filtered',classes_num_filtered)
        result = []
        for i in range(len(boxes_filtered)):
            result.append(
                [self.classes[classes_num_filtered[i]],
                 boxes_filtered[i][0],
                 boxes_filtered[i][1],
                 boxes_filtered[i][2],
                 boxes_filtered[i][3],
                 probs_filtered[i]])

        return result

    def iou(self, box1, box2):  # box接受[x,y,w,h]四维
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
             max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
             max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        inter = 0 if tb < 0 or lr < 0 else tb * lr
        return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)

    def draw_result(self, img, result):  # result接受[class,x,y,w,h,p]
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)  # （左上角，右下角）对角线坐标、color、粗细
            cv2.rectangle(img, (x - w, y - h - 20),
                          (x + w, y - h), (125, 125, 125), -1)  # 画上面一点点的框
            lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA  # cv2.LINE_AA 抗锯齿，使线条更光滑
            # 在图片上添加文字：（图像，内容，位置，字体，大小，颜色粗细，lineType）
            cv2.putText(
                img, result[i][0] + ' : %.2f' % result[i][5],
                (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, lineType)



