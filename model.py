import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from omegaconf import OmegaConf
from datasets import load_dataset
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
import colored_traceback.auto
from transformers import Wav2Vec2Model
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import numpy as np
import mediapipe as mp
import lpips
from torch.utils.checkpoint import checkpoint
import math
import random


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, batch_size):
        return self.input.repeat(batch_size, 1, 1, 1)

class AdaIN(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = nn.Linear(style_dim, in_channel * 2)

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)
        return gamma * out + beta

class StyleConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, upsample=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channel, out_channel, kernel_size, padding=kernel_size // 2
        )
        self.adain = AdaIN(out_channel, style_dim)
        self.upsample = upsample

    def forward(self, input, style):
        out = self.conv(input)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.adain(out, style)
        return out

class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.conv = nn.Conv2d(in_channel, 3, 1)
        self.adain = AdaIN(3, style_dim)

    def forward(self, input, style, skip=None):
        out = self.conv(input)
        out = self.adain(out, style)

        if skip is not None:
            if self.upsample:
                skip = F.interpolate(skip, scale_factor=2, mode='bilinear', align_corners=False)
            out = out + skip

        return out

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class StyleGANGenerator(nn.Module):
    def __init__(self, style_dim=6144, n_mlp=8, max_resolution=256):
        super().__init__()
        
        self.style_dim = style_dim  # 3 * 2048 (3 ResNet50 features)
        self.max_resolution = max_resolution

        # Mapping network
        layers = []
        for i in range(n_mlp):
            layers.append(nn.Linear(style_dim, style_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64
        }

        # Initial input processing
        self.input = nn.Linear(style_dim, 4 * 4 * self.channels[4])
        self.conv1 = StyleConv(self.channels[4], self.channels[4], 3, style_dim)
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(max_resolution, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channel = self.channels[4]
        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            self.convs.append(StyleConv(in_channel, out_channel, 3, style_dim, upsample=True))
            self.convs.append(StyleConv(out_channel, out_channel, 3, style_dim))
            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel

    def forward(self, features, resolution=None, noise=None, return_latents=False):
        if resolution is None:
            resolution = self.max_resolution
        
        styles = self.style(features)

        # Initial processing
        out = self.input(styles).view(-1, self.channels[4], 4, 4)
        
        if noise is not None:
            out = out + noise.reshape(out.shape[0], 1, 1, 1) * torch.randn_like(out)
        
        out = self.conv1(out, styles)
        skip = self.to_rgb1(out, styles)

        for i, (conv1, conv2, to_rgb) in enumerate(zip(self.convs[::2], self.convs[1::2], self.to_rgbs)):
            out = conv1(out, styles)
            if noise is not None:
                out = out + noise.reshape(out.shape[0], 1, 1, 1) * torch.randn_like(out)
            out = conv2(out, styles)
            if noise is not None:
                out = out + noise.reshape(out.shape[0], 1, 1, 1) * torch.randn_like(out)
            skip = to_rgb(out, styles, skip)

            if out.shape[-1] == resolution:
                break

        image = skip

        if return_latents:
            return image, styles
        else:
            return image


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, downsample=False):
        super().__init__()
        self.conv = spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding))
        self.downsample = downsample

    def forward(self, input):
        out = self.conv(input)
        if self.downsample:
            out = F.avg_pool2d(out, 2)
        return out

class StyleGANDiscriminator(nn.Module):
    def __init__(self, max_resolution=1024, channel_multiplier=2):
        super().__init__()
        
        self.max_resolution = max_resolution
        
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier
        }

        self.from_rgb = nn.ModuleDict()
        for res in self.channels.keys():
            self.from_rgb[str(res)] = ConvBlock(3, self.channels[res], 1, 0)

        self.convs = nn.ModuleList()

        log_size = int(math.log(max_resolution, 2))

        in_channel = self.channels[max_resolution]

        for i in range(log_size, 2, -1):
            out_channel = self.channels[2 ** (i - 1)]
            self.convs.append(ConvBlock(in_channel, out_channel, 3, 1))
            self.convs.append(ConvBlock(out_channel, out_channel, 3, 1, downsample=True))
            in_channel = out_channel

        self.final_conv = ConvBlock(in_channel, self.channels[4], 3, 1)
        self.final_linear = nn.Sequential(
            spectral_norm(nn.Linear(self.channels[4] * 4 * 4, 1)),
        )

    def forward(self, input, resolution=None):
        if resolution is None:
            resolution = self.max_resolution
        
        out = self.from_rgb[str(resolution)](input)

        for conv in self.convs:
            out = conv(out)
            if out.shape[-1] == 4:
                break

        out = self.final_conv(out)
        out = out.view(out.shape[0], -1)
        out = self.final_linear(out)

        return out



class IRFD(nn.Module):
    def __init__(self, max_resolution=256):
        super(IRFD, self).__init__()
        
        # Encoders (keeping ResNet50 backbones)
        self.Ei = self._create_encoder()  # Identity encoder
        self.Ee = self._create_encoder()  # Emotion encoder
        self.Ep = self._create_encoder()  # Pose encoder
        

        # CIPSGenerator-based generator
        self.Gd = StyleGANGenerator(style_dim=2048*3,max_resolution=max_resolution)  # 2048*3 because we're concatenating 3 encoder outputs
        
        # StyleGAN Discriminator
        self.D = StyleGANDiscriminator(max_resolution=max_resolution)
        
        self.Cm = nn.Linear(2048, 8)  # 8 = num_emotion_classes
        
        self.max_resolution = max_resolution
        self.current_resolution = max_resolution


    def adjust_for_resolution(self,resolution):
        self.current_resolution = resolution



    def _create_encoder(self):
        encoder = resnet50(pretrained=True)
        return nn.Sequential(*list(encoder.children())[:-1])
    
    def _prepare_generator_input(self, *features):
        # Flatten and concatenate the encoder outputs
        return torch.cat([f.view(f.size(0), -1) for f in features], dim=1)
    
    def forward(self, x_s, x_t, noise):
        # Encode source and target images
        fi_s = checkpoint(self.Ei, x_s)
        fe_s = checkpoint(self.Ee, x_s)
        fp_s = checkpoint(self.Ep, x_s)
        
        fi_t = checkpoint(self.Ei, x_t)
        fe_t = checkpoint(self.Ee, x_t)
        fp_t = checkpoint(self.Ep, x_t)

        
        # Randomly swap one type of feature (keeping this functionality)
        swap_type = torch.randint(0, 3, (1,)).item()
        if swap_type == 0:
            fi_s, fi_t = fi_t, fi_s
        elif swap_type == 1:
            fe_s, fe_t = fe_t, fe_s
        else:
            fp_s, fp_t = fp_t, fp_s
        
        # Prepare generator inputs
        gen_input_s = self._prepare_generator_input(fi_s, fe_s, fp_s)
        gen_input_t = self._prepare_generator_input(fi_t, fe_t, fp_t)
        
        # Generate reconstructed images using StyleGANGenerator
        x_s_recon = self.Gd(gen_input_s, self.current_resolution, noise=noise)
        x_t_recon = self.Gd(gen_input_t, self.current_resolution, noise=noise)
        
        
        # Apply softmax to emotion predictions
        emotion_pred_s = torch.softmax(self.Cm(fe_s.view(fe_s.size(0), -1)), dim=1)
        emotion_pred_t = torch.softmax(self.Cm(fe_t.view(fe_t.size(0), -1)), dim=1)
      
        return x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, emotion_pred_s, emotion_pred_t


# StyleGAN-specific loss functions
class StyleGANLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.criterion = nn.MSELoss()

    def forward(self, real, fake):
        return self.criterion(fake, torch.ones_like(fake)) + self.criterion(real, torch.zeros_like(real))



#  LogSumExp Trick for Numerical Stability:
eps = 1e-8  # Small epsilon value

def safe_div(numerator, denominator):
    return numerator / (denominator + eps)

def safe_log(x):
    return torch.log(x + eps)

def stable_softmax(x):
    shifted_x = x - x.max(dim=1, keepdim=True)[0]
    Z = torch.sum(torch.exp(shifted_x), dim=1, keepdim=True)
    return torch.exp(shifted_x) / Z

def stable_cross_entropy(logits, targets):
    num_classes = logits.size(1)
    one_hot_targets = F.one_hot(targets, num_classes=num_classes)
    stable_probs = stable_softmax(logits)
    return -torch.sum(one_hot_targets * safe_log(stable_probs), dim=1).mean()

def clip_loss(loss, min_val=-100, max_val=100):
    return torch.clamp(loss, min=min_val, max=max_val)

class ScaledMSELoss(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        return self.mse(input * self.scale, target * self.scale)

class IRFDLoss(nn.Module):
    def __init__(self, alpha=0.1, lpips_weight=0.3, landmark_weight=1.0, emotion_weight=1.0, identity_weight=1.0):
        super(IRFDLoss, self).__init__()
        self.alpha = alpha
        self.lpips_weight = lpips_weight
        self.landmark_weight = landmark_weight
        self.emotion_weight = emotion_weight
        self.identity_weight = identity_weight
        self.l2_loss = ScaledMSELoss(scale=0.1)
        self.ce_loss = stable_cross_entropy
        self.lpips_loss = lpips.LPIPS(net='vgg')

        # Initialize MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    def detect_landmarks(self, images):
        batch_size = images.size(0)
        landmarks_batch = []

        for i in range(batch_size):
            image = images[i]
            image_np = (image.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            results = self.face_mesh.process(image_np)
            if results.multi_face_landmarks:
                landmarks = np.array([(lm.x, lm.y, lm.z) for lm in results.multi_face_landmarks[0].landmark])
                landmarks_batch.append(torch.tensor(landmarks, device=images.device))
            else:
                landmarks_batch.append(None)

        return landmarks_batch

    def landmark_loss(self, x_real, x_recon):
        landmarks_real = self.detect_landmarks(x_real)
        landmarks_recon = self.detect_landmarks(x_recon)
        
        losses = []
        for lm_real, lm_recon in zip(landmarks_real, landmarks_recon):
            if lm_real is not None and lm_recon is not None:
                losses.append(F.mse_loss(lm_real, lm_recon))
            else:
                losses.append(torch.tensor(0.0, device=x_real.device))
        
        return torch.stack(losses).mean()

    def identity_loss(self, fi_s, fi_t):
        return self.l2_loss(fi_s, fi_t) * self.identity_weight

    def emotion_loss(self, fe_s, fe_t, emotion_labels_s, emotion_labels_t):
        emotion_pred_s = torch.softmax(fe_s, dim=1)
        emotion_pred_t = torch.softmax(fe_t, dim=1)
        loss_s = self.ce_loss(emotion_pred_s, emotion_labels_s)
        loss_t = self.ce_loss(emotion_pred_t, emotion_labels_t)
        return (loss_s + loss_t) * self.emotion_weight

    def pose_loss(self, fp_s, fp_t):
        return self.l2_loss(fp_s, fp_t) * self.landmark_weight

    def reconstruction_loss(self, x_s, x_t, x_s_recon, x_t_recon):
        l_self = self.l2_loss(x_s, x_s_recon) + self.l2_loss(x_t, x_t_recon)
        l_lpips = (self.lpips_loss(x_s, x_s_recon).mean() + 
                   self.lpips_loss(x_t, x_t_recon).mean()) * self.lpips_weight
        return l_self + l_lpips

    def forward(self, x_s, x_t, x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, emotion_labels_s, emotion_labels_t):
        l_pose_landmark = self.pose_loss(fp_s, fp_t) + self.landmark_loss(x_s, x_s_recon) + self.landmark_loss(x_t, x_t_recon)
        l_emotion = self.emotion_loss(fe_s, fe_t, emotion_labels_s, emotion_labels_t)
        l_identity = self.identity_loss(fi_s, fi_t)
        l_recon = self.reconstruction_loss(x_s, x_t, x_s_recon, x_t_recon)

        return l_pose_landmark, l_emotion, l_identity, l_recon

    def __del__(self):
        # Clean up MediaPipe resources
        self.face_mesh.close()