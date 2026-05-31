import torch
import numpy as np
from sympy import false
import utils
import os
import torch.nn.functional as F
import time



def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)
###########################################################################
def pad_to_specific_size(x, target_height, target_width):
    _, _, img_h, img_w = x.shape
    pad_h_total = target_height - img_h
    pad_w_total = target_width - img_w

    # 确保填充尺寸为正
    pad_h_total = max(pad_h_total, 0)
    pad_w_total = max(pad_w_total, 0)

    pad_h1 = pad_h_total // 2
    pad_h2 = pad_h_total - pad_h1
    pad_w1 = pad_w_total // 2
    pad_w2 = pad_w_total - pad_w1

    x_padded = F.pad(x, (pad_w1, pad_w2, pad_h1, pad_h2), mode='reflect')
    return x_padded, pad_h1, pad_w1
###########################################################################


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def parameters(self):
        # 返回模型参数
        return []

    def restore(self, val_loader):
        image_folder = os.path.join(self.args.image_folder, self.config.data.val_dataset)

###########################################################################
        total_time = 0
        num_images = 0
        first_image_processed = False
###########################################################################
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x_cond = x[:, :3, :, :].to(self.diffusion.device)
                b, c, h, w = x_cond.shape
                 # ==================== padding ====================
                J = 1                
                wave = 'bior4.4'      

                if wave == 'bior4.4':
                    pad_dwt, dec_len = 8, 10
                elif wave == 'bior2.2':
                    pad_dwt, dec_len = 4, 6
                else:
                    raise ValueError("Unsupported wavelet configured")

                C = 2 * pad_dwt - dec_len  

                tmp_h = img_h
                for _ in range(J):
                    tmp_h = (tmp_h + C) // 2 + 1  
                out_h = ((tmp_h + 15) // 16) * 16  
                target_height = out_h
                for _ in range(J):
                    target_height = (target_height - 1) * 2 - C

                tmp_w = img_w
                for _ in range(J):
                    tmp_w = (tmp_w + C) // 2 + 1
                out_w = ((tmp_w + 15) // 16) * 16
                target_width = out_w
                for _ in range(J):
                    target_width = (target_width - 1) * 2 - C
                x, pad_h, pad_w = pad_to_specific_size(x, target_height, target_width)

                out = self.model(x.to(self.device))
                pred_x = out["pred_x"]

                pred_x = pred_x[:, :, pad_h:pad_h + img_h, pad_w:pad_w + img_w]
                utils.logging.save_image(pred_x, os.path.join(image_folder, str(step), f"{y[0]}.png"))

                x_cond, pad_h, pad_w = pad_to_specific_size(x_cond, target_height, target_width)

###########################################################################
                start_time = time.time()
###########################################################################
                x_output = self.diffusive_restoration(x_cond)
                pad_h1 = pad_h
                pad_w1 = pad_w
                x_output = x_output[:, :, pad_h1:pad_h1 + h, pad_w1:pad_w1 + w]
###########################################################################
                #record the end time
                end_time = time.time()
                processing_time = end_time - start_time
                if first_image_processed:
                    total_time += processing_time
                    num_images += 1
                else:
                    first_image_processed = True
                utils.logging.save_image(x_output, os.path.join(image_folder, f"{y[0]}.png"))
###########################################################################
                print(f"Processing image {y[0]}, Time taken: {processing_time:.4f} seconds")
        average_time = total_time / num_images if num_images > 0 else 0
        print(f"Average time per image: {average_time:.4f} seconds")
########################################################################


    def diffusive_restoration(self, x_cond):
        x_output = self.diffusion.model(x_cond)
        return x_output["pred_x"]

