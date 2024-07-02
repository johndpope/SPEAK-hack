import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from tqdm.auto import tqdm
import os
from omegaconf import OmegaConf
from datasets import load_dataset
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
import colored_traceback.auto
from transformers import Wav2Vec2Model
import torch.nn.functional as F

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim, n_mlp):
        super().__init__()
        layers = [nn.Linear(latent_dim, latent_dim)]
        layers.extend([nn.Linear(latent_dim, latent_dim) for _ in range(n_mlp - 1)])
        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        return self.mapping(x)

class StyleGANGenerator(nn.Module):
    def __init__(self, latent_dim, n_mlp, channels=32):
        super().__init__()
        self.mapping = MappingNetwork(latent_dim, n_mlp)
        self.input = nn.Parameter(torch.randn(1, channels, 4, 4))
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.to_rgb1 = nn.Conv2d(channels, 3, 1)
        self.layers = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        for i in range(3, 10):  # 8x8 to 512x512
            in_channel = channels
            out_channel = channels
            self.layers.append(nn.Conv2d(in_channel, out_channel, 3, padding=1))
            self.layers.append(nn.Conv2d(out_channel, out_channel, 3, padding=1))
            self.to_rgbs.append(nn.Conv2d(out_channel, 3, 1))

    def forward(self, styles, noise=None):
        styles = [self.mapping(s) for s in styles]
        out = self.input.repeat(styles[0].shape[0], 1, 1, 1)
        out = self.conv1(out)
        skip = self.to_rgb1(out)
        for i, (conv1, conv2, to_rgb) in enumerate(zip(self.layers[::2], self.layers[1::2], self.to_rgbs)):
            out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
            out = conv1(out)
            out = conv2(out)
            skip = to_rgb(out) + F.interpolate(skip, scale_factor=2, mode='bilinear', align_corners=False)
        return skip

class IRFD(nn.Module):
    def __init__(self, latent_dim=512, n_mlp=8):
        super(IRFD, self).__init__()
        
        # Encoders (keeping ResNet50 backbones)
        self.Ei = self._create_encoder()  # Identity encoder
        self.Ee = self._create_encoder()  # Emotion encoder
        self.Ep = self._create_encoder()  # Pose encoder
        
        # StyleGAN-based generator
        self.Gd = StyleGANGenerator(latent_dim, n_mlp)

        self.Cm = nn.Linear(2048, 8)  # 8 = num_emotion_classes

    def _create_encoder(self):
        encoder = resnet50(pretrained=True)
        return nn.Sequential(*list(encoder.children())[:-1])
    
    def forward(self, x_s, x_t):
        # Encode source and target images
        fi_s = self.Ei(x_s)
        fe_s = self.Ee(x_s)
        fp_s = self.Ep(x_s)
        
        fi_t = self.Ei(x_t)
        fe_t = self.Ee(x_t)
        fp_t = self.Ep(x_t)
        
        # Randomly swap one type of feature (keeping this functionality)
        swap_type = torch.randint(0, 3, (1,)).item()
        if swap_type == 0:
            fi_s, fi_t = fi_t, fi_s
        elif swap_type == 1:
            fe_s, fe_t = fe_t, fe_s
        else:
            fp_s, fp_t = fp_t, fp_s
        
        # Concatenate features for generator input
        gen_input_s = torch.cat([fi_s, fe_s, fp_s], dim=1).squeeze(-1).squeeze(-1)
        gen_input_t = torch.cat([fi_t, fe_t, fp_t], dim=1).squeeze(-1).squeeze(-1)
        
        # Generate reconstructed images using StyleGAN-based generator
        x_s_recon = self.Gd([gen_input_s])
        x_t_recon = self.Gd([gen_input_t])
        
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


class IRFDLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(IRFDLoss, self).__init__()
        self.alpha = alpha
        self.l2_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, x_s, x_t, x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, emotion_pred_s, emotion_pred_t, emotion_labels_s, emotion_labels_t):
        # Ensure all images have the same size
        x_s = F.interpolate(x_s, size=x_s_recon.shape[2:], mode='bilinear', align_corners=False)
        x_t = F.interpolate(x_t, size=x_t_recon.shape[2:], mode='bilinear', align_corners=False)

# Ensure all inputs have the same batch size
        batch_size = x_s.size(0)
        
        # Reshape emotion predictions and labels if necessary
        emotion_pred_s = emotion_pred_s.view(batch_size, -1)
        emotion_pred_t = emotion_pred_t.view(batch_size, -1)
        emotion_labels_s = emotion_labels_s.view(batch_size)
        emotion_labels_t = emotion_labels_t.view(batch_size)
        # Identity loss
        l_identity = torch.max(
            self.l2_loss(fi_s, fi_t) - self.l2_loss(fi_s, fi_s) + self.alpha,
            torch.tensor(0.0).to(fi_s.device)
        )
        
        # Classification loss
        l_cls = self.ce_loss(emotion_pred_s, emotion_labels_s) + self.ce_loss(emotion_pred_t, emotion_labels_t)
        
        # Pose loss
        l_pose = self.l2_loss(fp_s, fp_t)
        
        # Emotion loss
        l_emotion = self.l2_loss(fe_s, fe_t)
        
        # Self-reconstruction loss
        l_self = self.l2_loss(x_s, x_s_recon) + self.l2_loss(x_t, x_t_recon)
        
        # Total loss
        total_loss = l_identity + l_cls + l_pose + l_emotion + l_self
        
        return total_loss, l_identity, l_cls, l_pose, l_emotion, l_self