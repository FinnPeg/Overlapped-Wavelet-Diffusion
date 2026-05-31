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

    # 初始化DWT变换
    dwt = DWTForward(J=1, wave='bior4.4', mode='reflect').cuda()
    # dwt = DWTForward(J=1, wave='bior2.2', mode='reflect').cuda()

    # 执行DWT
    yl, yh = dwt(img)

    # 提取低频子带 LL
    LL = yl

    # 提取高频子带 LH, HL, HH
    LH = yh[0][:, :, 0, :, :]  # 水平高频子带
    HL = yh[0][:, :, 1, :, :]  # 垂直高频子带
    HH = yh[0][:, :, 2, :, :]  # 对角高频子带

    # 调整通道顺序为 (n, c, h, w)
    x_LL = LL.permute(0, 1, 2, 3)  # LL 子带
    x_LH = LH.permute(0, 1, 2, 3)  # LH 子带
    x_HL = HL.permute(0, 1, 2, 3)  # HL 子带
    x_HH = HH.permute(0, 1, 2, 3)  # HH 子带

    # x_LL = x_LL[:, :, :h // 2, :w // 2]
    # x_LH = x_LH[:, :, :h // 2, :w // 2]
    # x_HL = x_HL[:, :, :h // 2, :w // 2]
    # x_HH = x_HH[:, :, :h // 2, :w // 2]

    # 打印子带形状（可选）
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
    # 创建零张量用于 LH、HL 和 HH
    # LH = torch.zeros_like(LL)
    # HL = torch.zeros_like(LL)
    # HH = torch.zeros_like(LL)

    # 计算需要填充的尺寸
    # pad_h = 1
    # pad_w = 1

    # 恢复子带填充
    # LL = F.pad(LL, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
    # LH = F.pad(LH, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
    # HL = F.pad(HL, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
    # HH = F.pad(HH, (pad_w, pad_w, pad_h, pad_h), mode='reflect')

    # 打印调试信息，检查每个子带的形状
    # print(f"LL shape: {LL.shape}")
    # print(f"LH shape: {LH.shape}")
    # print(f"HL shape: {HL.shape}")
    # print(f"HH shape: {HH.shape}")


    iwt = DWTInverse(wave='bior4.4', mode='reflect').cuda()
    # iwt = DWTInverse(wave='bior2.2', mode='reflect').cuda()
    # 组合高频子带
    yh = torch.stack((LH, HL, HH), dim=2)

    # 执行IDWT
    reconstructed_img = iwt((LL, [yh]))
    # 打印重建后的图像形状（可选）
    # print("Reconstructed image shape:", reconstructed_img.shape)
    # print("reconstructed_img range:", reconstructed_img.min(), reconstructed_img.max())

    # 反归一化处理
    # reconstructed_img = inverse_data_transform(reconstructed_img)
    # print("reconstructed_img_RGB range:", reconstructed_img.min(), reconstructed_img.max())
    # reconstructed_img = reconstructed_img * 255.0
    # print("reconstructed_img_RGB range:", reconstructed_img.min(), reconstructed_img.max())
    # reconstructed_img = torch.clamp(reconstructed_img, 0, 255)  # 确保数据范围在0到255之间
    # print("reconstructed_img_RGB range:", reconstructed_img.min(), reconstructed_img.max())
    # reconstructed_img = reconstructed_img.cpu().numpy().astype(np.uint8)
    # reconstructed_img = reconstructed_img.transpose(0, 2, 3, 1)

    return reconstructed_img


def shift_and_normalize(img, global_min, global_max):
    if img.ndim == 4:
        b, h, w, c = img.shape
        normalized_img_batch = np.zeros_like(img, dtype=np.uint8)
        for i in range(b):
            single_img = img[i].astype(np.float32)
            single_img = 2 * (single_img - global_min) / (global_max - global_min) - 1
            single_img = inverse_data_transform(single_img).astype(np.float32)
            single_img *= 255
            single_img = np.clip(single_img, 0, 255)
            # single_img = single_img - 128 + np.mean(single_img)
            normalized_img_batch[i] = single_img.astype(np.uint8)
        return normalized_img_batch
    elif img.ndim == 3:
        h, w, c = img.shape
        single_img = img.astype(np.float32)
        single_img = 2 * (single_img - global_min) / (global_max - global_min) - 1
        single_img = inverse_data_transform(torch.tensor(single_img)).numpy().astype(np.float32)
        single_img *= 255
        single_img = np.clip(single_img, 0, 255)
        # single_img = single_img - 128 + np.mean(single_img)
        return single_img.astype(np.uint8)
    else:
        raise ValueError("Unsupported image shape")


def combine_subbands(LL, HL, LH, HH):
    b, c, H, W = LL.shape
    # 去掉每个子带中多余的2行和2列，使子带恢复到原始图像大小的一半
    LL = LL[:, :, 2:H-2, 2:W-2]  # 裁剪掉上下和左右边缘
    HL = HL[:, :, 2:H-2, 2:W-2]
    LH = LH[:, :, 2:H-2, 2:W-2]
    HH = HH[:, :, 2:H-2, 2:W-2]

    # 重新计算裁剪后的高度和宽度
    H, W = LL.shape[2], LL.shape[3]
    print(LL.shape)

    combined_img_batch = np.zeros((b, 2 * H, 2 * W, 3), dtype=np.uint8)

    # 找到 HL, LH, HH 的全局最小值和最大值
    global_min = min(np.min(HL.cpu().numpy()), np.min(LH.cpu().numpy()), np.min(HH.cpu().numpy()))
    global_max = max(np.max(HL.cpu().numpy()), np.max(LH.cpu().numpy()), np.max(HH.cpu().numpy()))

    for i in range(b):
        LL_np = LL[i].cpu().numpy().transpose(1, 2, 0)
        HL_np = HL[i].cpu().numpy().transpose(1, 2, 0)
        LH_np = LH[i].cpu().numpy().transpose(1, 2, 0)
        HH_np = HH[i].cpu().numpy().transpose(1, 2, 0)

        # LL_norm = cv2.normalize(LL_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        LL_norm = np.clip(LL_np, 0, 255).astype(np.uint8)
        HL_norm = shift_and_normalize(HL_np, global_min, global_max)
        LH_norm = shift_and_normalize(LH_np, global_min, global_max)
        HH_norm = shift_and_normalize(HH_np, global_min, global_max)

        if HL_norm.ndim == 2:
            HL_norm = cv2.cvtColor(HL_norm, cv2.COLOR_GRAY2BGR)
        if LH_norm.ndim == 2:
            LH_norm = cv2.cvtColor(LH_norm, cv2.COLOR_GRAY2BGR)
        if HH_norm.ndim == 2:
            HH_norm = cv2.cvtColor(HH_norm, cv2.COLOR_GRAY2BGR)

        combined_img = np.zeros((2 * H, 2 * W, 3), dtype=np.uint8)
        combined_img[0:H, 0:W] = LL_norm
        combined_img[H:2 * H, 0:W] = HL_norm
        combined_img[0:H, W:2 * W] = LH_norm
        combined_img[H:2 * H, W:2 * W] = HH_norm

        combined_img_batch[i] = combined_img

    return combined_img_batch

def mse_loss(image_true, image_pred):
    """计算两张图片之间的均方误差（MSE）"""
    return np.mean((image_true.astype("float32") - image_pred.astype("float32")) ** 2)
#################################################################################################################
def main():
    data_path = '/home/user/Desktop/data/Diffusion-Low-Light/datasets/LOLv1/eval15/high_time'
    results_path = '/home/user/Desktop/data/Diffusion-Low-Light/results/HFRM/bior2.2_level1_nonblocknoise'

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    img_names = os.listdir(data_path)
    for img_name in img_names:
        img_path = os.path.join(data_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue  # Skip files that aren't images

        # Convert to 4D tensor (b, c, h, w)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).cuda()  # Convert to (b, c, h, w)


        # 进行DWT初始化
        combined_tensor = dwt_init(image)
        # 进行逆DWT并恢复图像
        reconstructed_img = iwt_init(combined_tensor)
        if reconstructed_img is not None:
            combined_image = combine_subbands(*torch.chunk(combined_tensor, 4, dim=0))[0]  # Convert to (h, w, c)

            output_file_path = os.path.join(results_path, f'bior_53DWT_{img_name}')
            cv2.imwrite(output_file_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

            output_file_path = os.path.join(results_path, f'bior_53IDWT_{img_name}')
            # 添加cpu().numpy()转换
            cv2.imwrite(output_file_path,
                        cv2.cvtColor(reconstructed_img[0].cpu().numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR))

            # 打印调试信息
            image_true_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy().astype("float32")
            reconstructed_img_np = reconstructed_img.squeeze(0).cpu().numpy().transpose(1, 2, 0).astype("float32")

            # print("image_true shape:", image_true_np.shape, "dtype:", image_true_np.dtype)
            # print("image_pred shape:", image_pred_np.shape, "dtype:", image_pred_np.dtype)
            # print("image_true range:", image_true_np.min(), image_true_np.max())
            # print("image_pred range:", reconstructed_img.min(), reconstructed_img.max())


            loss = mse_loss(image_true_np, reconstructed_img_np )
            print(f"Loss for {img_name}: {loss}\n\n")


if __name__ == "__main__":
    main()

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, image):
        return dwt_init(image)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, img):
        return iwt_init(img)

# class DWT_bior(nn.Module):
#     def __init__(self):
#         super(DWT_bior, self).__init__()
#         self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导
#
#     def forward(self, image):
#         return dwt_init(image)
#
#
# class IWT_bior(nn.Module):
#     def __init__(self):
#         super(IWT_bior, self).__init__()
#         self.requires_grad = False
#
#     def forward(self, img):
#         return iwt_init(img)









