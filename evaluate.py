from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2
import os
import pyiqa
import torch


# f_target = open("/home/user/Desktop/data/Diffusion-Low-Light/datasets/LSRW/val/LSRW_gt.txt").readlines()
# f_target = open("/home/user/Desktop/data/Diffusion-Low-Light/datasets/LOLv2/val/LOLv2_gt.txt").readlines()
# f_target = open("/home/user/Desktop/data/Diffusion-Low-Light/datasets/LOLv2/val/LOLv2_syn_gt.txt").readlines()
f_target = open("/home/user/Desktop/data/Diffusion-Low-Light/datasets/LOLv1/val/LOLv1_gt.txt").readlines()
f_target = list(map(lambda s: s.strip(), f_target))

psnr_total = 0
ssim_total = 0
lpips_total = 0
fid_total = 0

for i in range(0, len(f_target)):
    img1 = cv2.imread(f_target[i])
    _, f_result = os.path.split(f_target[i])
    print(f_result)
    # LOLv1
    f_result = os.path.join("/home/user/Desktop/data/Diffusion-Low-Light/results/test/LOLv1", f_result)
    # LOLv2
    # f_result = os.path.join("/home/user/Desktop/data/Diffusion-Low-Light/results/test/LOLv2", f_result)
    #LSRW
    # f_result = os.path.join("/home/user/Desktop/data/Diffusion-Low-Light/results/test/LSRW", f_result)
    img2 = cv2.imread(f_result)
    # print(img2.shape)
    # create metric with default setting
    lpips_metric = pyiqa.create_metric('lpips').cuda()
    # print(lpips_metric)

    restored = img1.transpose(2, 0, 1)
    target = img2.transpose(2, 0, 1)

    restored = torch.tensor(restored)
    target = torch.tensor(target)

    # restored = restored.to("cuda:0")
    # target = target.to("cuda:0")

    l = lpips_metric(restored, target)


    # img path as inputs.
    # l = lpips_metric(img1, img2)

    # For FID metric, use directory or precomputed statistics as inputs
    # refer to clean-fid for more details: https://github.com/GaParmar/clean-fid

    p = psnr(img1, img2)
    # print("Image shape:", img1.shape)
    s = ssim(img1, img2, channel_axis=-1) 


    print("PSNR:", p)
    print("SSIM:", s)
    print(f'LPIPS: { l:.4f}')
    print()

    psnr_total = p + psnr_total
    ssim_total = s + ssim_total
    lpips_total = l + lpips_total


    mean_psnr = psnr_total / len(f_target)
    mean_ssim = ssim_total / len(f_target)
    mean_lpips = lpips_total / len(f_target)

fid_metric = pyiqa.create_metric('fid')
# f = fid_metric('/home/user/Desktop/data/Diffusion-Low-Light/datasets/LSRW/test/high',
#                '/home/user/Desktop/data/Diffusion-Low-Light/results/test/LSRW')

f = fid_metric('/home/user/Desktop/data/Diffusion-Low-Light/datasets/LOLv1/eval15/high',
               '/home/user/Desktop/data/Diffusion-Low-Light/results/test/LOLv1')

# f = fid_metric('/home/user/Desktop/data/Diffusion-Low-Light/datasets/LOLv2/Real_captured/Test/high',
#                '/home/user/Desktop/data/Diffusion-Low-Light/results/test/LOLv2')

# f = fid_metric('/home/user/Desktop/data/Diffusion-Low-Light/datasets/LOLv2/Synthetic/Test/Normal',
#                '/home/user/Desktop/data/Diffusion-Low-Light/results/test/LOLv2')

print('The value of the average PSNR is:', mean_psnr)
print('The value of the average SSIM is:', mean_ssim)
print(f'The value of the average LPIPS is: { mean_lpips :.4f}')
print(f'FID : {f:.4f}')


