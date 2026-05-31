import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import utils
from models.unet import DiffusionUNet
from models.wavelet_bior44_torch import DWT, IWT
from pytorch_msssim import ssim
from models.mods_48 import HFRM
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import time
from skimage.metrics import peak_signal_noise_ratio

log_dir = '/home/user/Desktop/data/user/Diffusion-Low-Light/run_bior44_level1_HFE_ch48'
writer = SummaryWriter(log_dir)

def data_transform(X):
    return 2 * X - 1.0

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

##################################################################
def pad_to_specific_size(x, target_height, target_width):
    _, _, img_h, img_w = x.shape
    pad_h_total = target_height - img_h
    pad_w_total = target_width - img_w

    pad_h_total = max(pad_h_total, 0)
    pad_w_total = max(pad_w_total, 0)

    pad_h1 = pad_h_total // 2
    pad_h2 = pad_h_total - pad_h1
    pad_w1 = pad_w_total // 2
    pad_w2 = pad_w_total - pad_w1

    x_padded = F.pad(x, (pad_w1, pad_w2, pad_h1, pad_h2), mode='reflect')
    return x_padded, pad_h1, pad_w1
##################################################################

class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.depth_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=0, groups=in_ch)
        self.point_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.depth_conv(self.pad(x))
        x = self.point_conv(x)
        return x

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()

        self.args = args
        self.config = config
        self.device = config.device

        self.high_enhance0 = HFRM(in_channels=9, out_channels=9, wf=48)
        self.Unet = DiffusionUNet(config)
        self.ll_proj = Depth_conv(3, 48)

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_training(self, x_cond, b, eta=0.):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        n, c, h, w = x_cond.shape
        seq_next = [-1] + list(seq[:-1])
        x = torch.randn(n, c, h, w, device=self.device)
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            et = self.Unet(torch.cat([x_cond, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(x.device))

        return xs[-1]

    def forward(self, x, debug: bool = False):
        data_dict = {}
        dwt, idwt = DWT(), IWT()

        input_img = x[:, :3, :, :]
        n, c, h, w = input_img.shape
        input_img_norm = data_transform(input_img)

        input_dwt = dwt(input_img_norm)
        input_LL, input_high0 = input_dwt[:, :c, ...], input_dwt[:, c:, ...]

        b = self.betas.to(input_img.device)

        # t = torch.randint(low=0, high=self.num_timesteps, size=(input_LL_LL.shape[0] // 2 + 1,)).to(self.device)
        t = torch.randint(low=0, high=self.num_timesteps, size=(input_LL.shape[0] // 2 + 1,)).to(self.device)
        # t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:input_LL_LL.shape[0]].to(x.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:input_LL.shape[0]].to(x.device)
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

        # e = torch.randn_like(input_LL_LL)
        e = torch.randn_like(input_LL)

        if self.training:
            # gt_img_norm = x[:, 3:, :, :]
            gt_img_norm = data_transform(x[:, 3:, :, :])
            gt_dwt = dwt(gt_img_norm)
            gt_LL, gt_high0 = gt_dwt[:, :c, ...], gt_dwt[:, c:, ...]

            # gt_LL_dwt = dwt(gt_LL)
            # gt_LL_LL, gt_high1 = gt_LL_dwt[:n, ...], gt_LL_dwt[n:, ...]

            # x = gt_LL_LL * a.sqrt() + e * (1.0 - a).sqrt()
            x = gt_LL * a.sqrt() + e * (1.0 - a).sqrt()
            # noise_output = self.Unet(torch.cat([input_LL_LL, x], dim=1), t.float())

            noise_output = self.Unet(torch.cat([input_LL, x], dim=1), t.float())
            # denoise_LL_LL = self.sample_training(input_LL_LL, b)
            denoise_LL = self.sample_training(input_LL, b)

            input_high0 = self.high_enhance0(input_high0, denoise_LL)

            # pred_LL = idwt(torch.cat((denoise_LL_LL, input_high1), dim=0))

            pred_LL = idwt(torch.cat((denoise_LL, input_high0), dim=1))
            pred_x = pred_LL
            # pred_x = idwt(torch.cat((denoise_LL, input_high0), dim=0))
            pred_x = inverse_data_transform(pred_x)


            data_dict["input_high0"] = input_high0
            # data_dict["input_high1"] = input_high1
            data_dict["gt_high0"] = gt_high0
            # data_dict["gt_high1"] = gt_high1
            data_dict["pred_LL"] = pred_LL
            data_dict["gt_LL"] = gt_LL
            data_dict["noise_output"] = noise_output
            data_dict["pred_x"] = pred_x
            data_dict["e"] = e
            data_dict["denoise_LL"] = denoise_LL

        else:
            denoise_LL = self.sample_training(input_LL, b)
            input_high0 = self.high_enhance0(input_high0, denoise_LL)
            pred_LL = idwt(torch.cat((denoise_LL, input_high0), dim=1))
            pred_x = pred_LL
            pred_x = inverse_data_transform(pred_x)
            if debug:
                return {
                    "pred_x": pred_x,
                    "input_high0": input_high0,
                    "denoise_LL": denoise_LL,
                    "gt_LL": input_LL, 
                }
            else:
                return {"pred_x": pred_x}

        return data_dict


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = Net(args, config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.TV_loss = TVLoss()

        self.optimizer, self.scheduler = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
##################################################################
        #restart training
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
##################################################################
        if ema:
            self.ema_helper.ema(self.model)
        ####################################################
        print(f"=> loaded checkpoint '{load_path}' (epoch {self.start_epoch}, step {self.step})")
##################################################################

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()
        
        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            self.epoch = epoch
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)

                output = self.model(x)

                noise_loss, photo_loss, frequency_loss = self.estimation_loss(x, output)

                loss = noise_loss + photo_loss + frequency_loss

                if self.step % 10 == 0:
                    print("step:{}, lr:{:.6f}, noise_loss:{:.4f}, photo_loss:{:.4f}, "
                          "frequency_loss:{:.4f}".format(self.step, self.scheduler.get_last_lr()[0],
                                                         noise_loss.item(), photo_loss.item(),
                                                         frequency_loss.item()))


                # tensorboard
                writer.add_scalar("Train/noise_loss", noise_loss, self.step)
                writer.add_scalar("Train/photo_loss", photo_loss, self.step)
                writer.add_scalar("Train/frequency_loss", frequency_loss, self.step)
                writer.add_scalar("Train/total_loss", loss, self.step)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0 and self.step != 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step)
                    utils.logging.save_checkpoint({'step': self.step, 'epoch': epoch + 1,
                                                   'state_dict': self.model.state_dict(),
                                                   'optimizer': self.optimizer.state_dict(),
                                                   'scheduler': self.scheduler.state_dict(),
                                                   'ema_helper': self.ema_helper.state_dict(),
                                                   'params': self.args,
                                                   'config': self.config},
                                                  filename=os.path.join(self.config.data.ckpt_dir, 'model_latest'))

            self.scheduler.step()
        writer.close()

    def estimation_loss(self, x, output):
        input_high0, gt_high0 = output["input_high0"], output["gt_high0"]

        pred_LL, gt_LL, pred_x, noise_output, e, denoise_LL = output["pred_LL"], output["gt_LL"], output["pred_x"], \
            output["noise_output"], output["e"], output["denoise_LL"]

        denoise_LL = denoise_LL.to(self.device)
        pred_x = pred_x.to(self.device)

        gt_img = x[:, 3:, :, :].to(self.device)
        # =============noise loss==================
        noise_loss = self.l2_loss(noise_output, e)

        # =============frequency loss==================
        frequency_loss = 0.1 * ((self.l2_loss(input_high0, gt_high0) )+
                                self.l2_loss(denoise_LL, gt_LL)) + \
                         0.01 * (self.TV_loss(input_high0) +
                                   self.TV_loss(denoise_LL))

        # =============photo loss==================
        content_loss = self.l1_loss(pred_x, gt_img)

        ssim_loss = 1 - ssim(pred_x, gt_img, data_range=1.0).to(self.device)

        photo_loss = content_loss + ssim_loss

        return noise_loss, photo_loss, frequency_loss

    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.args.image_folder, self.config.data.type + str(self.config.data.patch_size))
        self.model.eval()
        total_val_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0

        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):
                b, _, img_h, img_w = x.shape
                gt_img = x[:, 3:, :, :].to(self.device)  
                # img_h_32 = int(32 * np.ceil(img_h / 32.0))
                # img_w_32 = int(32 * np.ceil(img_w / 32.0))
                # x = F.pad(x, (0, img_w_32 - img_w, 0, img_h_32 - img_h), 'reflect')
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

        # ======================== Best Model Save & Metric Print ========================
        if not hasattr(self, "best_psnr"):
            self.best_psnr = -1
            self.best_ssim = -1
            self.best_step_psnr = -1
            self.best_step_ssim = -1

        if avg_psnr > self.best_psnr:
            self.best_psnr = avg_psnr
            self.best_step_psnr = step
            save_path = f"/home/user/Desktop/data/user/Diffusion-Low-Light/bior22_HFE_ch48_level1_bestmodel/best_owdiff_{step}.pth.tar"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(
                {
                    'step': step,
                    'epoch': self.epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict() if hasattr(self, "optimizer") else None,
                    'scheduler': self.scheduler.state_dict() if hasattr(self, "scheduler") else None,
                    'ema_helper': self.ema_helper.state_dict() if hasattr(self, "ema_helper") else None,
                    'params': self.args if hasattr(self, "args") else None,
                    'config': self.config if hasattr(self, "config") else None,
                    'psnr': avg_psnr,
                    'ssim': avg_ssim
                },
                save_path
            )
            print(f"  PSNR: {avg_psnr:.4f}     Best: {self.best_psnr:.4f} @ {self.best_step_psnr} step")
            print(f"  SSIM: {avg_ssim:.4f}     Best: {self.best_ssim:.4f} @ {self.best_step_ssim} step")
            print(f"  >>>> Best model updated at step {step} (epoch {self.epoch})")

        return avg_val_loss





