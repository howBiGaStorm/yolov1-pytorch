# -*- coding: utf-8 -*-

import shutil  # 高级的文件操作模块
from collections import OrderedDict  # 使字典的Key是有序的
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
writer = SummaryWriter('./log')

import data
from yolov1.loss_layer import *
from yolov1.model import YOLO
from data.data_tf import *
import torchvision.models as models  # 子模块为一些构造好的模型
import torch.nn.init as init

def xavier(param):
    init.xavier_uniform(param)

def weight_init(m):
    if isinstance(m,nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def train(args):
    print('Dataset of instance(s) and batch size is {}'.format(args.batch_size))
    vgg = models.vgg16(False)  # 采用vgg16预训练好的模型
    vgg.load_state_dict(torch.load('vgg16-397923af.pth'))
    model = YOLO(vgg.features)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best = 1e+5
    print('加载模型是OK的')
    for epoch in range(1, args.epochs+1):
        l = train_epoch(epoch, model, optimizer, args)  # 返回一个epoch的平均损失

        # upperleft, bottomright, classes, confs = test_epoch(model, jpg='../data/1.jpg')
        is_best = l < best  # 如果损失小于到目前为止最好的损失，则标识符为TURE
        best = min(l, best)  # 更新到目前为止最好的损失
        # 每跑完一代，保存一次检查点
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best)
    # 训练完毕，提取最好的模型的状态字典
    checkpoint = torch.load('./model_best.pth.tar')
    state_dict = checkpoint['state_dict']

    # new_state_dict = OrderedDict()
    #
    # for k, v in state_dict.items():  #
    #     name = k[7:]  # 从第七层之后的keys
    #     new_state_dict[name] = v

    model.load_state_dict(state_dict)
    model.cpu()
    # model.cuda()
    torch.save(model.state_dict(), 'model_cpu.pth.tar')


def train_epoch(epoch, model, optimizer, args):
    losses = 0.0
    dataset = VOCdetection(root = VOC_ROOT)
    dataloader = data.DataLoader(dataset,args.batch_size,shuffle=False)
    criteria = myloss(batch_size=args.batch_size)
    i = 0
    batch_num = 0
    for (img, gt,width,height) in dataloader:
        img = Variable(img)
        labels = Variable(gt)

        optimizer.zero_grad()  # 优化器零梯度
        y_pred = model(img)  # 数据输入模型得输出



        l,a,b,c,d = criteria(y_pred, labels)  # 输出与真实值算损失

        l.backward()  # 损失反向传
        optimizer.step()  # 优化器走起来
        losses += l.data[0]
        batch_num += 1
        writer.add_scalar('loss',l.data.numpy(),batch_num)
        if l.data.numpy() < 1e-4:
            break
        print('No.{} batch || loss is {}'.format(batch_num,l.data[0]))
    print("Epoch: {}, Ave loss: {}".format(epoch, losses / batch_num))
    writer.close()
    return losses / batch_num

def test_epoch(model, use_cuda=False, jpg=None):
    if jpg is None:
        x = torch.randn(1, 3, 480, 640)
    else:
        img = plt.imread(jpg) / 255.
        x = torch.from_numpy(np.transpose(img, (2, 0, 1)))

    x = Variable(x, requires_grad=False)

    if use_cuda:
        x = x.cuda()

    y = model(x)
    upperleft, bottomright, classes, confs = convert2viz(y)


def pretrain():
    pass

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copy(filename, 'model_best.pth.tar')



