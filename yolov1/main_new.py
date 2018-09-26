
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
from yolov1.test import *
from tensorboardX import SummaryWriter
writer = SummaryWriter()

def xavier(param):
    init.xavier_uniform(param)

def weight_init(m):
    if isinstance(m,nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def main():
    parser = argparse.ArgumentParser(description='PyTorch YOLO')

    parser.add_argument('--use_cuda', type=bool, default=True,
                        help='use cuda or not')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Epochs')
    parser.add_argument('--batch_size', type=int, default=30,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')  # 随机种子

    args = parser.parse_args()

    # if torch.cuda.is_available():
    #     print(True)




    print('begin loading model!')
    vgg = models.vgg16(False)  # 采用vgg16预训练好的模型
    vgg.load_state_dict(torch.load('vgg16-397923af.pth'))
    model = YOLO(vgg.features)
    # model = YOLO(None)
    # model.apply(weight_init)
    # model.cuda()
    # model.yolo.apply(weight_init)

    model.cuda()
    # model.cpu()
    print('model load is complete!')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    dataset = VOCdetection(root=VOC_ROOT)
    dataloader = data.DataLoader(dataset, args.batch_size, shuffle=False)


    criteria = myloss(batch_size=args.batch_size).cuda()
    best_loss = 100.0
    for epoch in range(args.epochs):
        batch_num = 0
        losses = 0.0
        for (img, gt, height, width) in dataloader:
            img = Variable(img).cuda()
            # labels = Variable(gt)
            labels = Variable(gt).cuda()
            optimizer.zero_grad()  # 优化器零梯度
            y_pred = model(img)  # 数据输入模型得输出
            # print(y_pred)
            # y_pred = y_pred.data[0].numpy().reshape(1,1470)
            # allloss, a, b, c, d = loss_layer(y_pred, labels, batch_size=args.batch_size)  # 输出与真实值算损失
            # sum1 = np.array([a, b, c, d])
            # loss = [np.sum(sum1)]
            # l = torch.FloatTensor(loss)
            l,a,b,c,d = criteria(y_pred,labels)
            # print(l)

            # l1,a1,b1,c1,d1 = loss_layer(y_pred,labels,args.batch_size)
            # print(l1)

            losses += l.data.cpu().numpy()


            l.backward()  # 损失反向传
            optimizer.step()  # 优化器走起来
            # losses += l.data[0].numpy()
            batch_num += 1
            writer.add_scalar('loss',l,batch_num)

            if l.data.cpu().numpy()<best_loss:
                torch.save(model.state_dict(),'my_model_param.pth.tar')
                best_loss = l.data.cpu().numpy()
            if l.data.cpu().numpy()< 1e-3:
                break
            print('epoch.{}____>No.{} batch || loss is {},{},{},{},{}'.format(epoch,batch_num, l.data[0],a.data[0],b.data[0],c.data[0],d.data[0]))
        if losses < best_loss:
            torch.save(model.state_dict(), './all_trainval_epoch_'+str(epoch)+'.pkl')
            best_loss = losses
        print('epoch.{}____>Average loss is .{}'.format(epoch,losses/batch_num))
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # torch.save(model.state_dict(),'./my_model_epoch_'+str(epoch)+'.pkl')
        # print(b.data[0])
    # imname = '/home/neec10601/桌面/yolo-pytorch-master/data/VOCdevkit_small/VOC2007t/JPEGImages/000001.jpg'
    # img = cv2.imread(imname)
    # img_h, img_w, _ = img.shape  # 图片大小
    # image_size = 448
    # inputs = cv2.resize(img, (image_size, image_size))  # 图像缩放
    # inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)  # 调整图片频道，BGR -> RGB
    # inputs = (inputs / 255.0) * 2.0 - 1.0  # 调整像素值 在-1,1之间分布
    # inputs = np.reshape(inputs, (1, image_size, image_size, 3))
    # inputs = np.transpose(inputs, (0, 3, 1, 2))
    # inputs = Variable(torch.FloatTensor(inputs))
    # net_output = model(inputs)
    # # output = y_pred.data.numpy().reshape(1470)
    # output = net_output.data.numpy().reshape(1470)
    #
    # detect = Detector()
    # result = detect.interpret_output(output)
    # for i in range(len(result)):
    #     result[i][1] *= (1.0 * width / image_size)
    #     result[i][2] *= (1.0 * height / image_size)
    #     result[i][3] *= (1.0 * width / image_size)
    #     result[i][4] *= (1.0 * height / image_size)
    # detect.draw_result(img,result)
    # cv2.imshow('Image', img)
    # cv2.waitKey(0)
    writer.close()
    # print("Epoch: {}, Ave loss: {}".format(epoch, losses / batch_num))


if __name__ == '__main__':
    main()
