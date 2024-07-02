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
from torch.nn.utils import spectral_norm
import numpy as np

class ResNetEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        resnet = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, output_dim)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class MappingNetwork(nn.Module):
    def __init__(self, input_dim, style_dim, n_layers=8):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(n_layers):
            layers.extend([
                spectral_norm(nn.Linear(dim, dim)),
                nn.LeakyReLU(0.2)
            ])
        self.net = nn.Sequential(*layers, nn.Linear(dim, style_dim))
    
    def forward(self, x):
        return self.net(x)

class AdaIN(nn.Module):
    def __init__(self, feature_dim, style_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(feature_dim)
        self.style_scale = nn.Linear(style_dim, feature_dim)
        self.style_bias = nn.Linear(style_dim, feature_dim)
    
    def forward(self, x, style):
        normalized = self.instance_norm(x)
        scale = self.style_scale(style).unsqueeze(2).unsqueeze(3)
        bias = self.style_bias(style).unsqueeze(2).unsqueeze(3)
        return normalized * scale + bias


# dont use
class StyleGANGenerator(nn.Module):
    def __init__(self, style_dim, n_channels, max_resolution):
        super().__init__()
        self.const_input = nn.Parameter(torch.randn(1, n_channels, 4, 4))
        self.style_blocks = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        
        in_channels = n_channels
        resolution = 4
        
        while resolution <= max_resolution:
            self.style_blocks.append(nn.ModuleList([
                AdaIN(in_channels, style_dim),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                AdaIN(in_channels, style_dim),
                nn.Conv2d(in_channels, in_channels * 2, 3, padding=1),
            ]))
            self.to_rgb.append(nn.Conv2d(in_channels * 2, 3, 1))
            in_channels *= 2
            resolution *= 2
    
    def forward(self, w, alpha=1.0):
        x = self.const_input.repeat(w.size(0), 1, 1, 1)
        
        for i, (adain1, conv1, adain2, conv2) in enumerate(self.style_blocks):
            x = adain1(x, w)
            x = F.leaky_relu(conv1(x), 0.2)
            x = adain2(x, w)
            x = F.leaky_relu(conv2(x), 0.2)
            
            if i == len(self.style_blocks) - 1:
                return self.to_rgb[i](x)
            
            if i == len(self.style_blocks) - 2:
                upsample = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                rgb = self.to_rgb[i](x)
                rgb_next = self.to_rgb[i+1](upsample)
                return (1 - alpha) * rgb + alpha * rgb_next
            
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)


class FourierFeatures(nn.Module):
    def __init__(self, input_dim, mapping_size=256, scale=10):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.B = nn.Parameter(torch.randn((input_dim, mapping_size)) * scale, requires_grad=False)
    
    def forward(self, x):
        x = x.matmul(self.B)
        return torch.sin(x)

class ModulatedFC(nn.Module):
    def __init__(self, in_features, out_features, style_dim):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.modulation = nn.Linear(style_dim, in_features)
        
    def forward(self, x, style):
        style = self.modulation(style).unsqueeze(1)
        x = self.fc(x * style)
        return x

class CIPSGenerator(nn.Module):
    def __init__(self, input_dim, ngf=64, max_resolution=256, style_dim=512, num_layers=8):
        super(CIPSGenerator, self).__init__()
        
        self.input_dim = input_dim
        self.ngf = ngf
        self.max_resolution = max_resolution
        self.style_dim = style_dim
        self.num_layers = num_layers
        
        self.mapping_network = nn.Sequential(
            nn.Linear(input_dim, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim)
        )
        
        self.fourier_features = FourierFeatures(2, 256)
        self.coord_embeddings = nn.Parameter(torch.randn(1, 512, 256, 256))
        
        self.layers = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        
        current_dim = 512 + 256  # Fourier features + coordinate embeddings
        
        for i in range(num_layers):
            self.layers.append(ModulatedFC(current_dim, ngf * 8, style_dim))
            if i % 2 == 0 or i == num_layers - 1:
                self.to_rgb.append(ModulatedFC(ngf * 8, 3, style_dim))
            current_dim = ngf * 8
        
    def get_coord_grid(self, batch_size, resolution):
        x = torch.linspace(-1, 1, resolution)
        y = torch.linspace(-1, 1, resolution)
        x, y = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack((x, y), dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        return coords.to(next(self.parameters()).device)
    
    def forward(self, x, target_resolution):
        batch_size = x.size(0)
        
        # Map input to style vector
        w = self.mapping_network(x)
        
        # Generate coordinate grid
        coords = self.get_coord_grid(batch_size, target_resolution)
        coords_flat = coords.view(batch_size, -1, 2)
        
        # Get Fourier features and coordinate embeddings
        fourier_features = self.fourier_features(coords_flat)
        coord_embeddings = F.grid_sample(
            self.coord_embeddings.expand(batch_size, -1, -1, -1),
            coords,
            mode='bilinear',
            align_corners=True
        ).permute(0, 2, 3, 1).reshape(batch_size, -1, 512)
        
        # Concatenate Fourier features and coordinate embeddings
        features = torch.cat([fourier_features, coord_embeddings], dim=-1)
        
        rgb = 0
        for i, (layer, to_rgb) in enumerate(zip(self.layers, self.to_rgb)):
            features = layer(features, w)
            features = F.leaky_relu(features, 0.2)
            
            if i % 2 == 0 or i == self.num_layers - 1:
                rgb = rgb + to_rgb(features, w)
        
        output = torch.sigmoid(rgb).view(batch_size, target_resolution, target_resolution, 3).permute(0, 3, 1, 2)
        
        # Ensure output is in [-1, 1] range
        output = (output * 2) - 1
        
        return output

class IRFD(nn.Module):
    def __init__(self, max_resolution=256):
        super(IRFD, self).__init__()
        
        # Encoders (keeping ResNet50 backbones)
        self.Ei = self._create_encoder()  # Identity encoder
        self.Ee = self._create_encoder()  # Emotion encoder
        self.Ep = self._create_encoder()  # Pose encoder
        
        # CIPSGenerator-based generator
        self.Gd = CIPSGenerator(input_dim=2048*3)  # 2048*3 because we're concatenating 3 encoder outputs
        
        # StyleGAN Discriminator
        self.D = StyleGANDiscriminator(max_resolution, int(np.log2(max_resolution)) - 1)
        
        self.Cm = nn.Linear(2048, 8)  # 8 = num_emotion_classes

    def _create_encoder(self):
        encoder = resnet50(pretrained=True)
        return nn.Sequential(*list(encoder.children())[:-1])
    
    def _prepare_generator_input(self, *features):
        # Flatten and concatenate the encoder outputs
        return torch.cat([f.view(f.size(0), -1) for f in features], dim=1)
    
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
        
        # Prepare generator inputs
        gen_input_s = self._prepare_generator_input(fi_s, fe_s, fp_s)
        gen_input_t = self._prepare_generator_input(fi_t, fe_t, fp_t)
        
        # Generate reconstructed images using CIPSGenerator
        x_s_recon = self.Gd(gen_input_s, 256)
        x_t_recon = self.Gd(gen_input_t, 256)
        
        # Apply softmax to emotion predictions
        emotion_pred_s = torch.softmax(self.Cm(fe_s.view(fe_s.size(0), -1)), dim=1)
        emotion_pred_t = torch.softmax(self.Cm(fe_t.view(fe_t.size(0), -1)), dim=1)
      
        return x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, emotion_pred_s, emotion_pred_t


class StyleGANDiscriminator(nn.Module):
    def __init__(self, max_resolution, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        in_channels = 3
        resolution = max_resolution
        
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels, in_channels * 2, 4, 2, 1)),
                nn.LeakyReLU(0.2)
            ))
            in_channels *= 2
            resolution //= 2
            if resolution == 4:
                break
        
        self.final = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(in_channels, 1, 4, 1, 0))
        )
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final(x).squeeze()
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