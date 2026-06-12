import torch
import torch.nn as nn
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from pytorch_wavelets import DWTForward, DWTInverse
import os

def data_transform(X):
    return 2 * X - 1.0

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

def dwt_init(img):
    # print(img.shape, end='input_img.shape\n')
    # img = img.to(torch.float32).cuda()/255
    img = img.to(torch.float32).cuda()
    # img = data_transform(img)

    # b, c, h, w = img.shape
    # print(img.shape, end='input_img.shape\n')

    dwt = DWTForward(J=1, wave='bior4.4', mode='reflect').cuda()
    # dwt = DWTForward(J=1, wave='bior2.2', mode='reflect').cuda()

    yl, yh = dwt(img)

    LL = yl

    LH = yh[0][:, :, 0, :, :]  # 水平高频子带
    HL = yh[0][:, :, 1, :, :]  # 垂直高频子带
    HH = yh[0][:, :, 2, :, :]  # 对角高频子带

    x_LL = LL.permute(0, 1, 2, 3)  # LL 子带
    x_LH = LH.permute(0, 1, 2, 3)  # LH 子带
    x_HL = HL.permute(0, 1, 2, 3)  # HL 子带
    x_HH = HH.permute(0, 1, 2, 3)  # HH 子带

    # x_LL = x_LL[:, :, :h // 2, :w // 2]
    # x_LH = x_LH[:, :, :h // 2, :w // 2]
    # x_HL = x_HL[:, :, :h // 2, :w // 2]
    # x_HH = x_HH[:, :, :h // 2, :w // 2]

    # print("LL shape:", x_LL.shape)
    # print("LH shape:", x_LH.shape)
    # print("HL shape:", x_HL.shape)
    # print("HH shape:", x_HH.shape)

    return torch.cat((x_LL, x_LH, x_HL, x_HH), 0)
    # return torch.cat((x_LL, x_LH, x_HL, x_HH), 1)

def iwt_init(input_tensor):

    b, c, h, w =input_tensor.shape
    n = b // 4
    LL = input_tensor[:n, ...]
    LH = input_tensor[n:2 * n, ...]
    HL = input_tensor[2 * n:3 * n, ...]
    HH = input_tensor[3 * n:, ...]
    # c = c // 4
    # LL = input_tensor[:, :c, ...]
    # LH = input_tensor[:, c:2 * c, ...]
    # HL = input_tensor[:, 2 * c:3 * c, ...]
    # HH = input_tensor[:, 3 * c:, ...]

    iwt = DWTInverse(wave='bior4.4', mode='reflect').cuda()
    # iwt = DWTInverse(wave='bior2.2', mode='reflect').cuda()
    yh = torch.stack((LH, HL, HH), dim=2)

    reconstructed_img = iwt((LL, [yh]))

    return reconstructed_img


class DWT_bior(nn.Module):
    def __init__(self):
        super(DWT_bior, self).__init__()
        self.requires_grad = False 

    def forward(self, image):
        return dwt_init(image)


class IWT_bior(nn.Module):
    def __init__(self):
        super(IWT_bior, self).__init__()
        self.requires_grad = False

    def forward(self, img):
        return iwt_init(img)









