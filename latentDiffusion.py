import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
import copy
import csv
import torchvision.models as models
import time
import numpy as np
import random
from lossFunction import perceptual_loss_lpips_per_sample, fft_loss_per_sample, LaplacianLoss, PatchPerceptualLoss, contrast_loss, local_contrast_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def default_init(m):
    if isinstance(m, nn.Conv2d):
        if m.kernel_size == (1, 1):
            nn.init.xavier_uniform_(m.weight)
        else:
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def sinusoidal_embedding(timesteps, dim):
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device).float() / half)
    args = timesteps[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlockT(nn.Module):
    def __init__(self, in_ch, out_ch=None, time_dim=256, groups=8, use_conv_shortcut=False):
        super().__init__()
        out_ch = out_ch or in_ch
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2)
        )

        if in_ch != out_ch:
            if use_conv_shortcut:
                self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
            else:
                self.shortcut = nn.Sequential(
                    nn.GroupNorm(groups, in_ch),
                    nn.SiLU(),
                    nn.Conv2d(in_ch, out_ch, kernel_size=1)
                )
        else:
            self.shortcut = nn.Identity()

        default_init(self)

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        t_out = self.time_mlp(t_emb)
        scale, shift = t_out.chunk(2, dim=1)  # each (B, out_ch)

        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = self.act2(h)
        h = self.conv2(h)

        return self.shortcut(x) + h


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4, ffn_dim=2048):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

        self.ffn = nn.Sequential(
            nn.Conv2d(channels, ffn_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ffn_dim, channels, kernel_size=1)
        )

        self.ln1 = nn.LayerNorm(channels)
        self.ln2 = nn.LayerNorm(channels)

        default_init(self)

    def forward(self, x):
        B, C, H, W = x.shape

        x_ln = self.ln1(x.view(B, C, -1).permute(0, 2, 1)).permute(0, 2, 1).view(B, C, H, W)

        qkv = self.qkv(x_ln)  # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)

        def reshape_heads(t):
            return t.reshape(B, self.num_heads, C // self.num_heads, H * W)

        qh = reshape_heads(q)
        kh = reshape_heads(k)
        vh = reshape_heads(v)

        qh = qh / math.sqrt(C // self.num_heads)
        attn = torch.einsum("bhnc,bhmc->bhnm", qh, kh)  # (B, heads, N, N)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhnm,bhmc->bhnc", attn, vh)  # (B, heads, C//heads, N)
        out = out.reshape(B, C, H, W)
        out = self.proj(out)

        out = self.ln2(out.view(B, C, -1).permute(0, 2, 1)).permute(0, 2, 1).view(B, C, H, W)

        out = self.ffn(out)

        return out + x


class CrossAttention(nn.Module):
    def __init__(self, query_dim, cond_dim, num_heads=4, ffn_dim=None):
        super().__init__()
        assert query_dim % num_heads == 0
        self.num_heads = num_heads
        self.dim_head = query_dim // num_heads
        self.scale = math.sqrt(self.dim_head)

        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(cond_dim, query_dim)
        self.to_v = nn.Linear(cond_dim, query_dim)
        self.to_out = nn.Linear(query_dim, query_dim)

        if ffn_dim is None:
            ffn_dim = query_dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, query_dim)
        )

        self.norm_feat = nn.LayerNorm(query_dim)
        self.norm_cond = nn.LayerNorm(query_dim)
        self.norm_ffn = nn.LayerNorm(query_dim)

        default_init(self)

    def forward(self, feat, cond_tokens):
        B, C, H, W = feat.shape
        N = H * W

        feat = self.norm_feat(feat.view(B, C, N).permute(0, 2, 1))  # Normalize feat (query)
        cond_tokens = self.norm_cond(cond_tokens)  # Normalize cond_tokens (key/values)

        q = feat.view(B, C, N).permute(0, 2, 1)  # (B, N, C)

        q = self.to_q(q)  # (B, N, C)
        k = self.to_k(cond_tokens)  # (B, N_cond, C)
        v = self.to_v(cond_tokens)  # (B, N_cond, C)

        q = q.view(B, N, self.num_heads, self.dim_head).permute(0, 2, 1, 3)  # (B, heads, N, dh)
        k = k.view(B, -1, self.num_heads, self.dim_head).permute(0, 2, 3, 1)  # (B, heads, dh, N_cond)
        v = v.view(B, -1, self.num_heads, self.dim_head).permute(0, 2, 1, 3)  # (B, heads, N_cond, dh)

        attn = torch.matmul(q, k) / self.scale  # (B, heads, N, N_cond)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, heads, N, dh)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)  # (B, N, C)

        out = self.to_out(out)
        out = out + feat

        out = self.norm_ffn(out)

        out = self.ffn(out)

        out = out + feat

        # Reshape back to (B, C, H, W)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out


class CondImageEncoder(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, cond_dim=None, pretrained=True, freeze=True):
        super().__init__()
        if cond_dim is None:
            cond_dim = base_ch * 8
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # --- adapt first conv to in_ch ---
        old_conv1 = resnet.conv1
        new_conv1 = nn.Conv2d(in_ch,
                              old_conv1.out_channels,
                              kernel_size=old_conv1.kernel_size,
                              stride=1,
                              padding=old_conv1.padding,
                              bias=False)

        if pretrained:
            w = old_conv1.weight.data  # (out, 3, k, k)
            if in_ch == 1:
                new_conv1.weight.data = w.mean(dim=1, keepdim=True)
            elif in_ch == 3:
                new_conv1.weight.data = w.clone()
            else:
                new_conv1.weight.data = w[:, :1, ...].repeat(1, in_ch, 1, 1) / float(in_ch)
        else:
            nn.init.kaiming_normal_(new_conv1.weight, nonlinearity='relu')

        # store adapted conv1 as our own attribute
        self.new_conv1 = new_conv1

        self.bn1 = nn.GroupNorm(num_groups=8, num_channels=self.new_conv1.out_channels, affine=True)  # learnable params

        self.relu = resnet.relu
        self.maxpool = nn.Identity()  # disable maxpool

        # keep the ResNet blocks
        self.layer1 = self._replace_bn_with_gn(resnet.layer1)  # H
        self.layer2 = self._replace_bn_with_gn(resnet.layer2)  # H/2
        self.layer3 = self._replace_bn_with_gn(resnet.layer3)  # H/4
        self.layer4 = self._replace_bn_with_gn(resnet.layer4)  # H/8

        # projection convs to match your original encoder’s channel scheme
        self.proj1 = nn.Conv2d(64, base_ch, kernel_size=1)  # layer1
        self.proj2 = nn.Conv2d(128, base_ch * 2, kernel_size=1)  # layer2
        self.proj3 = nn.Conv2d(256, base_ch * 4, kernel_size=1)  # layer3
        self.proj4 = nn.Conv2d(512, cond_dim, kernel_size=1)  # layer4

        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = sobel_x.transpose(2, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        # optionally freeze backbone
        if freeze:
            for name, p in self.named_parameters():
                if "proj" in name or "new_conv1" in name:
                    p.requires_grad_(True)
                elif isinstance(p, nn.GroupNorm):
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)

        # save config
        self.pretrained = pretrained
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.cond_dim = cond_dim

    def _replace_bn_with_gn(self, layer):
        for name, module in layer.named_children():
            if isinstance(module, nn.BatchNorm2d):
                num_channels = module.num_features
                group_norm = nn.GroupNorm(num_groups=8, num_channels=num_channels, affine=True)
                setattr(layer, name, group_norm)
            elif isinstance(module, nn.Sequential):
                self._replace_bn_with_gn(module)
        return layer

    def local_contrast_norm(self, z, kernel_size=3, eps=1e-6):
        mean = F.avg_pool2d(z, kernel_size, stride=1, padding=kernel_size // 2)
        std = torch.sqrt(F.avg_pool2d((z - mean) ** 2, kernel_size, stride=1, padding=kernel_size // 2) + eps)
        z_norm = (z - mean) / (std + eps)
        return z_norm

    def contrast_residual(self, x, lam=0.5, ksize=5):
        # x: low-contrast image [B, 1, H, W]
        blur = F.avg_pool2d(x, kernel_size=ksize, stride=1, padding=ksize // 2)
        contrast_res = x + lam * (x - blur)
        contrast_res = torch.clamp(contrast_res, 0., 1.)
        return contrast_res

    def gradient_weight(self, f_c):
        gx = F.conv2d(f_c, self.sobel_x.expand(f_c.size(1), 1, 3, 3), padding=1, groups=f_c.size(1))
        gy = F.conv2d(f_c, self.sobel_y.expand(f_c.size(1), 1, 3, 3), padding=1, groups=f_c.size(1))
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        w = grad_mag / (grad_mag.mean(dim=(2, 3), keepdim=True) + 1e-6)
        return f_c * w

    def to_tokens(self, feat, proj):
        feat = proj(feat)
        B, C, H, W = feat.shape
        return feat.view(B, C, H * W).permute(0, 2, 1)  # (B, N, C)

    def forward(self, x):
        x01 = (x + 1.0) * 0.5
        x01 = self.local_contrast_norm(x01)
        x01 = self.contrast_residual(x01)

        f1 = self.relu(self.bn1(self.new_conv1(x01)))  # H
        f2 = self.layer1(f1)  # H
        f3 = self.layer2(f2)  # H/2
        f4 = self.layer3(f3)  # H/4
        f5 = self.layer4(f4)  # H/8

        return {
            "layer1": self.to_tokens(self.gradient_weight(f2), self.proj1),  # (B, N1, base_ch)
            "layer2": self.to_tokens(self.gradient_weight(f3), self.proj2),  # (B, N2, base_ch*2)
            "layer3": self.to_tokens(self.gradient_weight(f4), self.proj3),  # (B, N3, base_ch*4)
            "layer4": self.to_tokens(self.gradient_weight(f5), self.proj4),  # (B, N4, cond_dim)
        }

class MidBlockWithCrossAttn(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=256, cond_dim=None, num_heads=16, depth=4):
        super().__init__()
        self.depth = depth
        self.in_res = ResBlockT(in_channels, out_channels, time_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "self_attn": SelfAttention(out_channels, num_heads=num_heads),
                "cross_attn": CrossAttention(query_dim=out_channels, cond_dim=cond_dim, num_heads=num_heads),
                "res": ResBlockT(out_channels, out_channels, time_dim)
            }))

        default_init(self.in_res)
        for layer in self.layers:
            default_init(layer["res"])

    def forward(self, x, t_emb, cond_tokens_mid):
        h = self.in_res(x, t_emb)

        for layer in self.layers:
            h = layer["self_attn"](h)

            h = layer["cross_attn"](h, cond_tokens_mid)

            h = layer["res"](h, t_emb)

        return h


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=256, cond_dim=None, num_heads=8, depth=1):
        super().__init__()
        self.depth = depth
        self.in_res = ResBlockT(in_channels, out_channels, time_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "self_attn": SelfAttention(out_channels, num_heads=num_heads),
                "cross_attn": CrossAttention(query_dim=out_channels, cond_dim=cond_dim, num_heads=num_heads),
                "res": ResBlockT(out_channels, out_channels, time_dim)
            }))
        self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        default_init(self.in_res)
        for layer in self.layers:
            default_init(layer["res"])

    def forward(self, x, t_emb, cond_tokens):
        h = self.in_res(x, t_emb)

        for layer in self.layers:
            h = layer["self_attn"](h)

            h = layer["cross_attn"](h, cond_tokens)

            h = layer["res"](h, t_emb)

        return self.pool(h), h


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=256, cond_dim=None, num_heads=8, depth=1):
        super().__init__()
        self.depth = depth
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # Nearest-neighbor upsampling
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # Learnable refinement
        )
        self.in_res = ResBlockT(in_channels, out_channels, time_dim)
        self.skipCrossAtt = CrossAttention(query_dim=out_channels, cond_dim=cond_dim, num_heads=num_heads)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "self_attn": SelfAttention(out_channels, num_heads=num_heads),
                "res": ResBlockT(out_channels, out_channels, time_dim)
            }))
        default_init(self.in_res)
        default_init(self.skipCrossAtt)
        for layer in self.layers:
            default_init(layer["res"])

    def forward(self, x, skip, t_emb, cond_tokens):
        x = self.up(x)

        skip = self.skipCrossAtt(skip, cond_tokens)

        x = torch.cat([x, skip], dim=1)
        h = self.in_res(x, t_emb)  # expand channels

        for layer in self.layers:
            h = layer["self_attn"](h)
            h = layer["res"](h, t_emb)
        return h


class TinyUNet(nn.Module):
    def __init__(self, in_ch=16, cond_in_ch=1, base_ch=64, time_dim=256):  # cond_dim=512
        super().__init__()
        self.time_dim = time_dim

        # conditioning encoder
        self.cond_encoder = CondImageEncoder(in_ch=cond_in_ch, base_ch=base_ch, cond_dim=8 * base_ch)

        # time MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # input conv
        self.inc = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        # down path
        self.down1 = DownBlock(base_ch, base_ch, time_dim, cond_dim=base_ch, num_heads=2, depth=2)  # 64
        self.down2 = DownBlock(base_ch, base_ch * 2, time_dim, cond_dim=2 * base_ch, num_heads=4)  # 128
        self.down3 = DownBlock(base_ch * 2, base_ch * 4, time_dim, cond_dim=4 * base_ch, num_heads=8)  # 256

        # middle with cross-attn
        self.mid = MidBlockWithCrossAttn(base_ch * 4, base_ch * 8, time_dim, cond_dim=8 * base_ch, num_heads=16)  # 512

        # up path
        self.up3 = UpBlock(base_ch * 8, base_ch * 4, time_dim, cond_dim=4 * base_ch, num_heads=8) # 256
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, time_dim, cond_dim=2 * base_ch, num_heads=4)  # 128
        self.up1 = UpBlock(base_ch * 2, base_ch, time_dim, cond_dim=base_ch, num_heads=2, depth=2)  # 64

        # output
        self.out_conv = nn.Conv2d(base_ch, in_ch, kernel_size=1)

        self.init_weights()

    def init_weights(self):
        # Input conv
        default_init(self.inc)

        # Time MLP
        for m in self.time_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # Down / Up / Mid
        for block in [self.down1, self.down2, self.down3, self.mid, self.up3, self.up2, self.up1]:
            for sub in block.modules():
                default_init(sub)

        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x_t, cond_img, t):
        # prepare time embedding
        temb = sinusoidal_embedding(t, self.time_dim)
        temb = self.time_mlp(temb)

        # cond tokens
        cond_tokens = self.cond_encoder(cond_img)

        h0 = self.inc(x_t)  # (B, base_ch, H, W)
        x1, skip1 = self.down1(h0, temb, cond_tokens['layer1'])  # x1: base_ch/ pooled
        x2, skip2 = self.down2(x1, temb, cond_tokens['layer2'])  # x2: base_ch*2
        x3, skip3 = self.down3(x2, temb, cond_tokens['layer3'])  # x3: base_ch*4, h/8, w/8

        m = self.mid(x3, temb, cond_tokens['layer4'])  # (B, base_ch*4, H/8, W/8)

        u3 = self.up3(m, skip3, temb, cond_tokens['layer3'])  # (B, base_ch*2, H/4, W/4)
        u2 = self.up2(u3, skip2, temb, cond_tokens['layer2'])  # (B, base_ch, H/2, W/2)
        u1 = self.up1(u2, skip1, temb, cond_tokens['layer1'])  # (B, base_ch, H, W)

        out = self.out_conv(u1)
        return out

def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02, schedule="linear"):
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        target_final = 1e-4
        scale = (target_final / alphas_cumprod[-1]).sqrt()
        betas = 1 - (1 - betas) ** scale
        return betas

    elif schedule == "cosine":
        t = np.linspace(0, T, T + 1)
        alphas_cumprod = np.cos((t / T + 0.008) / 1.008 * np.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, 0, 0.01)
        return torch.tensor(betas, dtype=torch.float32)

    else:
        raise NotImplementedError(f"Unknown schedule: {schedule}")

class DiffusionNoiseSchedule:
    def __init__(self, T=1000, device="cpu"):
        self.T = T
        self.device = device
        betas = make_beta_schedule(T, schedule='linear').to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_om_ac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ac * x_start + sqrt_om_ac * noise

def p_sample(model, x_t, t, schedule, cond_img=None):
    betas_t = schedule.betas[t].view(-1, 1, 1, 1)
    sqrt_recip_alphas_t = schedule.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

    # predict epsilon
    if cond_img is None:
        eps_pred = model(x_t, t)
    else:
        eps_pred = model(x_t, cond_img, t)

    mean = sqrt_recip_alphas_t * (
        x_t - betas_t / (sqrt_one_minus_alphas_cumprod_t + 1e-12) * eps_pred
    )

    posterior_var = schedule.posterior_variance[t].view(-1, 1, 1, 1)

    noise = torch.randn_like(x_t) if (t[0] > 0) else torch.zeros_like(x_t)

    x_prev = mean + torch.sqrt(posterior_var) * noise
    return x_prev, mean


def cosine_ramp(start_w, end_w, current_epoch, start_epoch, end_epoch):
    if current_epoch < start_epoch:
        return start_w
    if current_epoch > end_epoch:
        return end_w
    t = (current_epoch - start_epoch) / (end_epoch - start_epoch)
    return start_w + 0.5 * (1 - math.cos(math.pi * t)) * (end_w - start_w)

def get_loss_weights(epoch):
    ddpm = 1.0
    x0_latent = 0.2
    recon_loss = 0.5

    # perc_loss = cosine_ramp(1.0, 2.0, epoch, 0, 10)
    # fft_loss = cosine_ramp(2.0, 3.0, epoch, 0, 10)
    # lap_loss = cosine_ramp(1.5, 2.0, epoch, 0, 10)
    # patch_perc_loss = cosine_ramp(0.0, 0.15, epoch, 10, 25)
    # contrast_loss_w = cosine_ramp(1.0, 2.0, epoch, 10, 25)
    # local_contrast_loss_w = cosine_ramp(0.3, 1.2, epoch, 10, 25)

    perc_loss = 1.0
    fft_loss = 2.0
    lap_loss = 1.5
    patch_perc_loss = 0.15
    contrast_loss_w = 6.6
    local_contrast_loss_w = 2.0


    return {
        "ddpm": ddpm,  # matching noise
        "x0_latent": x0_latent,  # L1 loss between x0_latent and x0_latent_bar
        "recon_loss": recon_loss,  # L1 loss between real images and generated images
        "perc_loss": perc_loss,  # perceptual features difference between real images and generated images, VGG16 based
        "fft_loss": fft_loss,  # frequency domain consistency
        "lap_loss": lap_loss,  # laplacian pyramid difference
        "patch_perc_loss": patch_perc_loss,  # patch-wise perceptual loss mobileNet based
        "contrast_loss": contrast_loss_w,  # image-wise standard deviation between prediction and target
        "local_contrast_loss": local_contrast_loss_w,  # aligning mean intensity and contrast (std) between predicted and ground-truth images
    }

def train_latent_ddpm(
        vae,
        unet,
        dataset,
        T=1000,
        device="cuda",
        epochs=100,
        batch_size=4,
        lr=1e-3,
        ckpt_dir="./ckpts_unet",
        use_ema=True,
        ema_beta=0.999,
        use_aux_recon_loss=True,
        grad_clip=1.0,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    schedule = DiffusionNoiseSchedule(T=T, device=device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=1e-4)

    batches_per_epoch = len(dataloader)
    total_steps = epochs * batches_per_epoch
    warmup_steps = 5000
    log_file_U, log_file_V = "training_log_UNet.csv", "Valid_log_Model.csv"
    if not os.path.exists(log_file_U):
        with open(log_file_U, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "avg_loss", "avg_ddpm", "avg_x0_latent", "avg_recon_aux", "avg_perc", "avg_fft", 'avg_lap', 'avg_patch_perc', "avg_contrast", "avg_local_contrast", "lr"])
    if not os.path.exists(log_file_V):
        with open(log_file_V, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "avg_loss", "avg_ddpm", "avg_x0_latent", "avg_recon_aux", "avg_perc", "avg_fft", 'avg_lap', 'avg_patch_perc', "avg_contrast", "avg_local_contrast", "lr"])

    def warmup_lambda(step):
        return min(1.0, step / warmup_steps)

    def get_condition_features(DSImg, truth, p_drop=0.1):
        if random.random() < p_drop:
            if random.random() < 0.5:
                cond_input = DSImg
            else:
                cond_input = torch.zeros_like(DSImg)
        else:
            cond_input = truth
        return cond_input

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(total_steps - warmup_steps),
        eta_min=5e-5
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    mse = nn.MSELoss(reduction='mean')
    l1 = nn.L1Loss(reduction='none')

    if use_ema:
        ema_unet = copy.deepcopy(unet).eval()
        ema_unet.to(device)
        for p in ema_unet.parameters(): p.requires_grad_(False)
    else:
        ema_unet = None

    bestLoss = float("inf")
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        loss_weights = get_loss_weights(epoch + 1)
        epoch_loss = 0
        total_x0_latent = 0
        total_recon_aux = 0
        total_perc = 0
        total_fft = 0
        total_ddpm = 0
        total_lap = 0
        total_patch_perc = 0
        total_contrast = 0
        total_local_contrast = 0

        for batch in pbar:
            DSImg, truth = batch[0], batch[1]
            truth = (truth.to(device) * 2.0 - 1.0)  # [-1,1]
            DSImg = (DSImg.to(device) * 2.0 - 1.0)  # [-1,1]

            with torch.no_grad():
                mu, logvar = vae.encoder(DSImg)  # (B,C,H,W)
                latents = vae.reparameterize(mu, logvar).detach()

            B = latents.shape[0]
            t = torch.randint(0, T, (B,), device=device).long()
            noise = torch.randn_like(latents, device=device)
            # forward q(z_t|z0)
            x_t = schedule.q_sample(latents, t, noise=noise)

            eps_pred = unet(x_t, get_condition_features(DSImg, truth), t)
            loss_ddpm = mse(eps_pred, noise) * loss_weights["ddpm"]
            total_ddpm += loss_ddpm.item()

            if use_aux_recon_loss:
                sqrt_ac = schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
                sqrt_om_ac = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

                x0_pred = (x_t - sqrt_om_ac * eps_pred) / (sqrt_ac + 1e-12)  # (B,C,H,W)
                loss_x0_latent = F.smooth_l1_loss(x0_pred, latents, reduction='mean') * loss_weights['x0_latent']

                recon_img = vae.decoder(x0_pred)

                truth_img = truth  # already [-1,1]
                x0_pred_img01 = (recon_img + 1) / 2  # b, c, 256, 256
                truth_img01 = (truth_img + 1) / 2

                if x0_pred_img01.size(1) == 1:
                    x0_pred_img01 = x0_pred_img01.repeat(1, 3, 1, 1)
                    truth_img01 = truth_img01.repeat(1, 3, 1, 1)

                lap_loss_fn = LaplacianLoss().to(device)
                patch_loss_fn = PatchPerceptualLoss(patch_size=16).to(device)
                loss_lap = lap_loss_fn(recon_img, truth_img) * loss_weights['lap_loss']
                loss_patch_perc = patch_loss_fn(x0_pred_img01, truth_img01) * loss_weights['patch_perc_loss']

                # L1 per-sample (image domain).
                l1_per_px = l1(recon_img, truth)  # (B,1,H,W)
                l1_per_sample = l1_per_px.view(B, -1).mean(dim=1)  # (B,)
                loss_recon_aux = l1_per_sample.mean() * loss_weights['recon_loss']

                # LPIPS per-sample
                lpips_per_sample = perceptual_loss_lpips_per_sample(x0_pred_img01, truth_img01)  # pass in [0,1]
                loss_perc = lpips_per_sample.mean() * loss_weights['perc_loss']

                # FFT per-sample
                fft_per_sample = fft_loss_per_sample(recon_img, truth)
                loss_fft = fft_per_sample.mean() * loss_weights['fft_loss']

                # contrast loss
                loss_contrast = contrast_loss(recon_img, truth) * loss_weights['contrast_loss']

                # local contrast loss
                loss_local_contrast = local_contrast_loss(recon_img, truth) * loss_weights['local_contrast_loss']

            else:
                loss_x0_latent = 0.
                loss_recon_aux = 0.
                loss_perc = 0.
                loss_fft = 0.
                loss_lap = 0.
                loss_patch_perc = 0.
                loss_contrast = 0.
                loss_local_contrast = 0.

            loss = loss_ddpm + loss_x0_latent + loss_recon_aux + loss_perc + loss_fft + loss_lap + loss_patch_perc + loss_contrast + loss_local_contrast

            optimizer.zero_grad()
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), grad_clip)

            optimizer.step()
            scheduler.step()

            if ema_unet is not None:
                update_ema(ema_unet, unet, beta=ema_beta)

            epoch_loss += loss.item()
            total_x0_latent += (loss_x0_latent if isinstance(loss_x0_latent, float) else loss_x0_latent.item())
            total_recon_aux += (loss_recon_aux if isinstance(loss_recon_aux, float) else loss_recon_aux.item())
            total_perc += (loss_perc if isinstance(loss_perc, float) else loss_perc.item())
            total_fft += (loss_fft if isinstance(loss_fft, float) else loss_fft.item())
            total_lap += (loss_lap if isinstance(loss_lap, float) else loss_lap.item())
            total_patch_perc += (loss_patch_perc if isinstance(loss_patch_perc, float) else loss_patch_perc.item())
            total_contrast += (loss_contrast if isinstance(loss_contrast, float) else loss_contrast.item())
            total_local_contrast += (loss_local_contrast if isinstance(loss_local_contrast, float) else loss_local_contrast.item())
            pbar.set_postfix(
                loss=loss.item(),
                ddpm=loss_ddpm.item(),
                x0_latent=(loss_x0_latent if isinstance(loss_x0_latent, float) else loss_x0_latent.item()),
                recon_aux=(loss_recon_aux if isinstance(loss_recon_aux, float) else loss_recon_aux.item()),
                perc=(loss_perc if isinstance(loss_perc, float) else loss_perc.item()),
                fft=(loss_fft if isinstance(loss_fft, float) else loss_fft.item()),
                lap=(loss_lap if isinstance(loss_lap, float) else loss_lap.item()),
                patch=(loss_patch_perc if isinstance(loss_patch_perc, float) else loss_patch_perc.item()),
                contrast=(loss_contrast if isinstance(loss_contrast, float) else loss_contrast.item()),
                local_contrast=(loss_local_contrast if isinstance(loss_local_contrast, float) else loss_local_contrast.item()),
            )

        avg_loss = epoch_loss / len(dataloader)
        avg_x0_latent = total_x0_latent / len(dataloader)
        avg_recon_aux = total_recon_aux / len(dataloader)
        avg_perc = total_perc / len(dataloader)
        avg_fft = total_fft / len(dataloader)
        avg_ddpm = total_ddpm / len(dataloader)
        avg_lap = total_lap / len(dataloader)
        avg_patch_perc = total_patch_perc / len(dataloader)
        avg_contrast = total_contrast / len(dataloader)
        avg_local_contrast = total_local_contrast / len(dataloader)
        print(f"Stats of UNet on training set: Epoch {epoch + 1}/{epochs} avg loss: {avg_loss:.4f} avg_ddpm: {avg_ddpm:.4f} avg_x0_latent: {avg_x0_latent:.4f} avg_recon_aux: {avg_recon_aux:.4f} avg_perc: {avg_perc:.4f} avg_fft: {avg_fft:.4f} lap: {avg_lap:.4f} patch: {avg_patch_perc:.4f} contrast: {avg_contrast:.4f} local contrast: {avg_local_contrast:.4f} lr: {scheduler.get_last_lr()[0]:.6f}")
        with open(log_file_U, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_loss, avg_ddpm, avg_x0_latent, avg_recon_aux, avg_perc, avg_fft, avg_lap, avg_patch_perc, avg_contrast, avg_local_contrast, scheduler.get_last_lr()[0]])

        if (epoch + 1) % 5 == 0:
            print('**********Validation Start**********')
            starTime = time.time()
            if ema_unet is not None:
                valid_loss = valid(ema_unet, schedule, T)
            else:
                valid_loss = valid(unet, schedule, T)
            print(f'Total validation Time: {time.time() - starTime:.2f} seconds')
            with open(log_file_V, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, valid_loss['avg loss'], valid_loss['avg ddpm'], valid_loss['avg x0 latent'], valid_loss['avg recon aux'], valid_loss['avg perc loss'], valid_loss['avg fft loss'], valid_loss['avg lap'], valid_loss['avg patch perc loss'], valid_loss['avg contrast loss'], valid_loss['avg local contrast loss'], scheduler.get_last_lr()[0]])

            if valid_loss['avg loss'] < bestLoss:
                bestLoss = valid_loss['avg loss']
                save_name = os.path.join(ckpt_dir, f"unet_best_epoch{epoch + 1}.pth")
                if ema_unet is not None:
                    torch.save(ema_unet.state_dict(), save_name)
                else:
                    torch.save(unet.state_dict(), save_name)
                print(f"Saved best UNet -> {save_name}")

            save_name = os.path.join(ckpt_dir, f"unet_checkPoint_epoch{epoch + 1}.pth")
            if ema_unet is not None:
                torch.save(ema_unet.state_dict(), save_name)
            else:
                torch.save(unet.state_dict(), save_name)
            print(f"Saved best UNet -> {save_name}")
            print('**********Validation Over**********')

    # final save
    final_name = os.path.join(ckpt_dir, "unet_final.pth")
    if ema_unet is not None:
        torch.save(ema_unet.state_dict(), final_name)
    else:
        torch.save(unet.state_dict(), final_name)
    print(f"Saved final UNet -> {final_name}")

@torch.no_grad()
def valid(model, schedule, T):
    dataset = ImageDataset('valid', device)
    dataloader = DataLoader(dataset, batch_size=8)
    mse = nn.MSELoss(reduction='mean')
    l1 = nn.L1Loss(reduction='none')
    weights = {
        "ddpm": 1.0,
        "x0_latent": 0.2,
        "recon_loss": 0.5,
        "perc_loss": 1.0,
        "fft_loss": 2.0,
        "lap_loss": 1.5,
        "patch_perc_loss": 0.15,
        "contrast_loss": 6.6,
        "local_contrast_loss": 2.0,
    }
    total_x0_latent = 0
    total_recon_aux = 0
    total_perc = 0
    total_fft = 0
    total_ddpm = 0
    total_lap = 0
    total_patch_perc = 0
    total_loss = 0
    total_contrast = 0
    total_local_contrast = 0
    for DSImg, truth in dataloader:
        truth = (truth.to(device) * 2.0 - 1.0)  # [-1,1]
        DSImg = (DSImg.to(device) * 2.0 - 1.0)  # [-1,1]

        mu, logvar = vae.encoder(DSImg)  # (B,C,H,W)
        std = torch.exp(0.5 * logvar)
        eps_z = torch.randn_like(std)
        latents = (mu + eps_z * std).detach()
        B = latents.shape[0]

        t = torch.randint(0, T, (B,), device=device).long()
        noise = torch.randn_like(latents, device=device)
        # forward q(z_t|z0)
        x_t = schedule.q_sample(latents, t, noise=noise)

        eps_pred = model(x_t, DSImg, t)
        loss_ddpm = mse(eps_pred, noise) * weights["ddpm"]

        sqrt_ac = schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_om_ac = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        x0_pred = (x_t - sqrt_om_ac * eps_pred) / (sqrt_ac + 1e-12)  # (B,C,H,W)
        loss_x0_latent = F.smooth_l1_loss(x0_pred, latents, reduction='mean') * weights['x0_latent']
        recon_img = vae.decoder(x0_pred)

        truth_img = truth  # already [-1,1]
        x0_pred_img01 = (recon_img + 1) / 2  # b, c, 256, 256
        truth_img01 = (truth_img + 1) / 2

        if x0_pred_img01.size(1) == 1:
            x0_pred_img01 = x0_pred_img01.repeat(1, 3, 1, 1)
            truth_img01 = truth_img01.repeat(1, 3, 1, 1)

        lap_loss_fn = LaplacianLoss().to(device)
        patch_loss_fn = PatchPerceptualLoss(patch_size=16).to(device)
        loss_lap = lap_loss_fn(recon_img, truth_img) * weights['lap_loss']
        loss_patch_perc = patch_loss_fn(x0_pred_img01, truth_img01) * weights['patch_perc_loss']

        # L1 per-sample (image domain). truth is [-1,1]
        l1_per_px = l1(recon_img, truth)  # (B,1,H,W)
        l1_per_sample = l1_per_px.view(B, -1).mean(dim=1)  # (B,)
        loss_recon_aux = l1_per_sample.mean() * weights['recon_loss']

        # LPIPS per-sample (requires channel-repeat inside function)
        lpips_per_sample = perceptual_loss_lpips_per_sample(x0_pred_img01, truth_img01)  # pass in [0,1]
        loss_perc = lpips_per_sample.mean() * weights['perc_loss']

        # FFT per-sample
        fft_per_sample = fft_loss_per_sample(recon_img, truth)
        loss_fft = fft_per_sample.mean() * weights['fft_loss']

        # contrast loss
        loss_contrast = contrast_loss(recon_img, truth) * weights['contrast_loss']

        # local contrast loss
        loss_local_contrast = local_contrast_loss(recon_img, truth) * weights['local_contrast_loss']

        loss = loss_ddpm + loss_x0_latent + loss_recon_aux + loss_perc + loss_fft + loss_lap + loss_patch_perc + loss_contrast + loss_local_contrast

        total_loss += loss.item()
        total_ddpm += loss_ddpm.item()
        total_x0_latent += loss_x0_latent.item()
        total_recon_aux += loss_recon_aux.item()
        total_perc += loss_perc.item()
        total_fft += loss_fft.item()
        total_lap += loss_lap.item()
        total_patch_perc += loss_patch_perc.item()
        total_contrast += loss_contrast.item()
        total_local_contrast += loss_local_contrast.item()
    avg_loss = total_loss / len(dataloader)
    avg_ddpm = total_ddpm / len(dataloader)
    avg_x0_latent = total_x0_latent / len(dataloader)
    avg_recon_aux = total_recon_aux / len(dataloader)
    avg_perc = total_perc / len(dataloader)
    avg_fft = total_fft / len(dataloader)
    avg_lap = total_lap / len(dataloader)
    avg_patch_perc = total_patch_perc / len(dataloader)
    avg_contrast = total_contrast / len(dataloader)
    avg_local_contrast = total_local_contrast / len(dataloader)
    print(f'Stats of model on validation set: avg loss: {avg_loss:.4f}, ddpm loss: {avg_ddpm:.4f}, x0 latent loss: {avg_x0_latent:.4f}, recon aux loss: {avg_recon_aux:.4f},'
          f' perc loss: {avg_perc:.4f}, fft loss: {avg_fft:.4f}, lap loss: {avg_lap:.4f}, patch perc loss: {avg_patch_perc:.4f}, contrast loss: {avg_contrast:.4f}, local contrast loss: {avg_local_contrast:.4f}')
    return {
        'avg loss': avg_loss,
        'avg ddpm': avg_ddpm,
        'avg x0 latent': avg_x0_latent,
        'avg recon aux': avg_recon_aux,
        'avg perc loss': avg_perc,
        'avg fft loss': avg_fft,
        'avg lap': avg_lap,
        'avg patch perc loss': avg_patch_perc,
        'avg contrast loss': avg_contrast,
        'avg local contrast loss': avg_local_contrast,
    }


@torch.no_grad()
def update_ema(ema_model, model, beta=0.999):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(beta).add_(p.data, alpha=1 - beta)

def sample_img2img_latent_ddpm(
        vae,
        unet,
        noisy_image,
        T=1000,
        start_t=50,
        device="cuda",
):
    unet.eval()
    vae.eval()
    noisy_image = (noisy_image * 2 - 1).to(device)  # normalize to [-1, 1]

    with torch.no_grad():
        mu, logvar = vae.encoder(noisy_image)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_noisy = mu + eps * std
        B = z_noisy.size(0)

        schedule = DiffusionNoiseSchedule(T=T, device=device)
        eps = torch.randn_like(z_noisy, device=device)
        sqrt_ac = schedule.sqrt_alphas_cumprod[start_t].view(1, 1, 1, 1)
        sqrt_om_ac = schedule.sqrt_one_minus_alphas_cumprod[start_t].view(1, 1, 1, 1)
        x_t = sqrt_ac * z_noisy + sqrt_om_ac * eps

        x_prev = x_t
        for time in range(start_t, -1, -1):
            t_batch = torch.full((B,), time, device=device, dtype=torch.long)
            x_prev, mean = p_sample(unet, x_prev, t_batch, schedule, cond_img=noisy_image)

        recon_final = vae.decoder(x_prev)
        recon_final = (recon_final + 1) / 2  # normalize to [0, 1]
        return recon_final


# ---- Utilities for saving .tif with zfill naming ----
def save_tensor_as_tif(img_tensor, path):
    x = img_tensor.detach().cpu().float()
    if x.min() < 0:
        x = (x + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)
    pil = to_pil_image(x)
    pil.save(path)


if __name__ == "__main__":
    from diffusionDataset import ImageDataset
    from VAE import VAE
    from huggingface_hub import hf_hub_download

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = VAE(in_channels=1, hidden_channels=256, latent_channels=16).to(device)
    vae_path = hf_hub_download(repo_id="James0323/enhanceImg", filename="VAESSIM.pth")
    vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # instantiate UNet
    unet = TinyUNet().to(device)
    # state_dict = torch.load("./ckpts_unet/unet_checkPoint_epoch25.pth", map_location=device, weights_only=True)
    # unet.load_state_dict(state_dict)
    unet.train()

    # ---- TRAIN ----
    train_ds = ImageDataset('train', device)
    train_latent_ddpm(
        vae=vae,
        unet=unet,
        dataset=train_ds,
        T=1000,
        device=device,
        epochs=100,
        batch_size=4,
        lr=1e-4,
        ckpt_dir="./ckpts_unet"
    )
