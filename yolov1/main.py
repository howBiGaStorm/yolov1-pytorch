# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


import argparse  # 命令行调用
from yolov1 import train


def main():
    parser = argparse.ArgumentParser(description='PyTorch YOLO')

    parser.add_argument('--use_cuda', type=bool, default=True,
                        help='use cuda or not')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Epochs')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')  # 随机种子

    args = parser.parse_args()

    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = args.use_cuda
    train.train(args)  # train文件中的train函数




if __name__ == '__main__':
    main()
