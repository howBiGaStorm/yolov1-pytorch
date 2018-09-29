import xml.etree.ElementTree as ET
import os.path as osp
import torch.utils.data as data
import torch
import cv2
import numpy as np
from collections import OrderedDict  # 使字典的Key是有序的


from torch.autograd import Variable
HOME = 'E:/grade_2_up\computer-Vision\yolo-pytorch-master'

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
        self.image_size= 448

    def __call__(self, target_root,height,width):

        labels = []
        for obj in target_root.iter('object'):
            bbox = obj.find('bndbox')
            x1 = max(min((float(bbox.find('xmin').text) - 1) , self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) , self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) , self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) , self.image_size - 1), 0)
            cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]

            bndbox = [x1/width,y1/height,x2/width,y2/height,cls_ind]
            labels +=[bndbox]

        return labels


class VOCdetection(data.Dataset):

    def __init__(self,root,pic_transform = None,target_transform = VOCAnnotationTransform()):
        self.root = root
        self.pic_transform = pic_transform
        self.target_transform = target_transform
        self.annopath = osp.join('%s','Annotations','%s.xml')
        self.imgpath = osp.join('%s','JPEGImages','%s.jpg')
        self.ids = list()
        self.image_size = 448
        VOC_2007t_ROOT = osp.join(self.root,'VOC2007t/')
        with open(VOC_2007t_ROOT+'ImageSets/Main/'+'eval.txt') as f:
            all_id = f.readlines()
        for id in range(len(all_id)):
            self.ids.append((VOC_2007t_ROOT,all_id[id].strip()))

    def __getitem__(self, item):
        img, gt = self.pull_item(item)
        return img, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self,item):
        path_and_id = self.ids[item]
        target_root = ET.parse(self.annopath % path_and_id)
        img = cv2.imread(self.imgpath % path_and_id)
        height, width, channel = img.shape

        gt_labels = self.target_transform(target_root, height, width)
        if self.pic_transform is not None:
            gtt = np.array(gt_labels)
            boxes = gtt[:, :4]
            labels = gtt[:, -1]
            img, labels = self.pic_transform(img, boxes, labels)
            img = img[:, :, (2, 1, 0)]
        return torch.from_numpy(img).permute(2, 0, 1), labels








