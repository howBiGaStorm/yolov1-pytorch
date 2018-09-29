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
        h_ratio = 1.0*self.image_size/ height
        w_ratio = 1.0*self.image_size/ width
        # print('h_ratio&w_ration:',h_ratio,w_ratio)
        label = np.zeros((S, S, 25))
        for obj in target_root.iter('object'):
            bbox = obj.find('bndbox')
            # coord = ['xmin','ymin','xmax','ymax']
            bndbox = []
            # name = obj.find('name').text.lower().strip()
            # class_num = self.class_to_ind[name]
            # x1 = (float(bbox.find('xmin').text) - 1) * w_ratio
            # y1 = (float(bbox.find('ymin').text) - 1) * h_ratio
            # x2 = (float(bbox.find('xmax').text) - 1) * w_ratio
            # y2 = (float(bbox.find('ymax').text) - 1) * h_ratio
            # print('*ratio:',x1,y1,x2,y2)
            # print((x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1)
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
            cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]

            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
            # for i,cord in enumerate(coord):
            #     coord_num = float(bbox.find(cord).text)-1
            #     if i%2 == 0:
            #         coord_num = max(min(coord_num * w_ratio,448-1),0)
            #     else:
            #         coord_num = max(min(coord_num * h_ratio,448-1),0)
            #     bndbox.append(coord_num)
            # boxes = [(bndbox[2]+bndbox[0])/2.0, bndbox[3]+bndbox[1]/2.0, bndbox[2]-bndbox[0], bndbox[3]-bndbox[1]]
            x_ind = int(boxes[0] * S / 448)
            y_ind = int(boxes[1] * S / 448)
            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            label[y_ind, x_ind, 5 + cls_ind] = 1
            labels = np.array(label,dtype=np.float32)
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

        image = cv2.resize(img,(self.image_size,self.image_size))

        gt_labels = self.target_transform(target_root, height, width)

        # image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)
        # image = (image/255.0) *2.0 -1.0

        # res = [xmin,ymin,xmax,ymax,class_num]

        # img_convert = torch.from_numpy(img).float().permute(2,0,1)
        img_convert = np.transpose(np.array(image, dtype=np.float32), (2, 0, 1))
        return img_convert, gt_labels




