import matplotlib.pyplot as plt
# import yolov1.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
# import sys
# sys.path.append('E:\grade_2_up\computer-Vision\yolo-pytorch-master')
from data.data1 import *

dataset = VOCdetection(root = VOC_ROOT)
dataloader = data.DataLoader(dataset=dataset,batch_size = 1,shuffle=False)
dataload = iter(dataloader)
target_list = []
w1 = []
h2=[]
batch_num = 0
for (img, gt,width,height) in dataloader:
    target_list.append(gt)
    w1.append(width)
    h2.append(height)
    batch_num += 1
print(img)
print(target_list[0]['class_probs'])
print(w1[0],h2[0])
print(batch_num)
# img,target = dataload
# print(dataload)



