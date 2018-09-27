import xml.etree.ElementTree as ET
import os.path as osp
import torch.utils.data as data
import torch
import cv2
import numpy as np
from collections import OrderedDict  # 使字典的Key是有序的
from torch.autograd import Variable
HOME = 'E:\grade_2_up\computer-Vision\yolo-pytorch-master'

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# labels = dict()
labels = OrderedDict(zip(VOC_CLASSES,range(len(VOC_CLASSES))))
# print(labels['bus'])

VOC_ROOT = osp.join(HOME, "data/VOCdevkit/")
S=7
B=2
C=20


class VOCAnnotationTransform(object):

    def __init__(self):
        self.class_to_ind = dict(zip(VOC_CLASSES,range(len(VOC_CLASSES))))

    def __call__(self, target_root,height,width):
        res = []
        # h_ratio = 1.0 * 448 / height
        # w_ratio = 1.0 * 448 / width
        for obj in target_root.iter('object'):
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            coord = ['xmin','ymin','xmax','ymax']
            bndbox = []
            class_num = self.class_to_ind[name]
            bndbox.append(class_num)
            for i,cord in enumerate(coord):
                coord_num = int(bbox.find(cord).text)-1
                # if i%2 == 0:
                #     coord_num *=h_ratio
                # else:
                #     coord_num *=w_ratio
                bndbox.append(coord_num)
            res.append(bndbox)
        cell_x = 1. * width / S  # width per cell
        cell_y = 1. * height / S  # height per cell
        for obj in res:
            center_x = 0.5 * (obj[1] + obj[3])  # (xmin + xmax) / 2
            center_y = 0.5 * (obj[2] + obj[4])  # (ymin + ymax) / 2
            cx = center_x / cell_x  # rescale the center x to cell size 表示在第几个格子里
            cy = center_y / cell_y  # rescale the center y to cell size
            if cx >= S or cy >= S: return None, None

            obj[3] = float(obj[3] - obj[1]) / width  # calculate and normalize width
            obj[4] = float(obj[4] - obj[2]) / height  # calculate and normalize height
            obj[3] = np.sqrt(obj[3])  # sqrt w
            obj[4] = np.sqrt(obj[4])  # sqrt h

            obj[1] = cx - np.floor(cx)  # center x in each cell 表示在这个格子的那个位置（百分比形式）
            obj[2] = cy - np.floor(cy)  # center x in each cell
            obj += [int(np.floor(cy) * S + np.floor(cx))]  # indexing cell[0, 49)  # 表示obj在49个格子中的哪一个

        class_probs = np.zeros([S * S, C])  # for one_hot vector per each cell 每个网格独热码[49,20]
        confs = np.zeros([S * S, B])  # for 2 bounding box per each cell 每个网格两个BBox的置信度 [49,2]
        coord = np.zeros([S * S, B, 4])  # for 4 coordinates per bounding box per cell 每个网格两个BBox的位置信息[49,2,4]
        proid = np.zeros([S * S, C])  # for class_probs weight \mathbb{1}^{obj}  每个网格类概率权重[49,20]
        prear = np.zeros([S * S, 4])  # for bounding box coordinates  每个网格一个BBox的位置信息[49,4]
        for obj in res:
            class_probs[obj[5], :] = [0.] * C  # no need?
            if not obj[0] in labels: continue

            class_probs[obj[5], obj[0]] = 1.  # 网格有对象的，类概率设为独热码
            confs[obj[5], :] = [1.] * B  # 网格里有目标的设为1

            # assign [center_x_in_cell, center_y_in_cell, w_in_image, h_in_image]
            coord[obj[5], :, :] = [obj[1:5]] * B

            # for 1_{i}^{obj} in paper eq.(3)
            proid[obj[5], :] = [1] * C

            # transform width and height to the scale of coordinates
            prear[obj[5], 0] = obj[1] - obj[3] ** 2 * 0.5 * S  # x_left
            prear[obj[5], 1] = obj[2] - obj[4] ** 2 * 0.5 * S  # y_top
            prear[obj[5], 2] = obj[1] + obj[3] ** 2 * 0.5 * S  # x_right
            prear[obj[5], 3] = obj[2] + obj[4] ** 2 * 0.5 * S  # y_bottom
        # for calculate upleft, bottomright and areas for 2 bounding box(not for 1 bounding box)
        upleft = np.expand_dims(prear[:, 0:2], 1)
        bottomright = np.expand_dims(prear[:, 2:4], 1)
        wh = bottomright - upleft
        area = wh[:, :, 0] * wh[:, :, 1]
        upleft = np.concatenate([upleft] * B, 1)
        bottomright = np.concatenate([bottomright] * B, 1)
        areas = np.concatenate([area] * B, 1)

        y_true = {
                'class_probs': np.array(class_probs,dtype=np.float32),
                'confs': np.array(confs,dtype=np.float32),
                'coord': np.array(coord,dtype=np.float32),
                'proid': np.array(proid,dtype=np.float32),
                'areas': np.array(areas,dtype=np.float32),
                'upleft': np.array(upleft,dtype=np.float32),
                'bottomright': np.array(bottomright,dtype=np.float32),
                }
        return y_true


class VOCdetection(data.Dataset):

    def __init__(self,root,target_transform = VOCAnnotationTransform()):
        self.root = root
        self.target_transform = target_transform
        self.annopath = osp.join('%s','Annotations','%s.xml')
        self.imgpath = osp.join('%s','JPEGImages','%s.jpg')
        self.ids = list()
        VOC_2007t_ROOT = osp.join(self.root,'VOC2007t/')
        with open(VOC_2007t_ROOT+'ImageSets/Main/'+'trainval.txt') as f:
            all_id = f.readlines()
        for id in range(len(all_id)):
            self.ids.append((VOC_2007t_ROOT,all_id[id].strip()))

    def __getitem__(self, item):
        img, gt, width, height = self.pull_item(item)
        return img, gt,width,height

    def __len__(self):
        return len(self.ids)

    def pull_item(self,item):
        path_and_id = self.ids[item]
        target_root = ET.parse(self.annopath % path_and_id)
        img = cv2.imread(self.imgpath % path_and_id)

        height,width,channel = img.shape
        img = cv2.resize(img, (448, 448))
        # res = [xmin,ymin,xmax,ymax,class_num]
        y_true = self.target_transform(target_root,height,width)
        # y_true = np.array(y_true,dtype=np.float32)
        # feed_gt = {key:torch.from_numpy(y_true[key]).float() for key in y_true.keys()}
        # img_convert = torch.from_numpy(img).float().permute(2,0,1)
        img_convert = np.transpose(np.array(img,dtype=np.float32),(2,0,1))
        return img_convert ,y_true ,height, width




