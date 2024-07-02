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
        self.coord_embeddings = nn.Parameter(torch.randn(1, 512, max_resolution, max_resolution))
        
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

class CIPSDiscriminator(nn.Module):
    def __init__(self, input_dim=3, ndf=64, max_resolution=256, num_layers=8):
        super(CIPSDiscriminator, self).__init__()
        
        self.input_dim = input_dim
        self.ndf = ndf
        self.max_resolution = max_resolution
        self.num_layers = num_layers
        
        self.fourier_features = FourierFeatures(2, 256)
        self.coord_embeddings = nn.Parameter(torch.randn(1, 512, max_resolution, max_resolution))
        
        self.layers = nn.ModuleList()
        
        current_dim = 512 + 256 + input_dim  # Fourier features + coordinate embeddings + input channels
        
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(current_dim, ndf * 8),
                nn.LeakyReLU(0.2)
            ))
            current_dim = ndf * 8
        
        self.final = nn.Linear(current_dim, 1)
    
    def get_coord_grid(self, batch_size, resolution):
        x = torch.linspace(-1, 1, resolution)
        y = torch.linspace(-1, 1, resolution)
        x, y = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack((x, y), dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        return coords.to(next(self.parameters()).device)
    
    def forward(self, x):
        batch_size, _, height, width = x.shape
        
        # Generate coordinate grid
        coords = self.get_coord_grid(batch_size, height)
        coords_flat = coords.view(batch_size, -1, 2)
        
        # Get Fourier features
        fourier_features = self.fourier_features(coords_flat)
        
        # Get coordinate embeddings
        if height != self.max_resolution or width != self.max_resolution:
            coord_embeddings = F.interpolate(
                self.coord_embeddings,
                size=(height, width),
                mode='bilinear',
                align_corners=True
            )
        else:
            coord_embeddings = self.coord_embeddings

        coord_embeddings = coord_embeddings.expand(batch_size, -1, -1, -1)
        coord_embeddings = coord_embeddings.permute(0, 2, 3, 1).reshape(batch_size, -1, 512)
        
        # Flatten input image
        x_flat = x.permute(0, 2, 3, 1).reshape(batch_size, -1, self.input_dim)
        
        # Concatenate all features
        features = torch.cat([x_flat, fourier_features, coord_embeddings], dim=-1)
        
        # Pass through layers
        for layer in self.layers:
            features = layer(features)
        
        # Final prediction
        output = self.final(features)
        
        return output.mean(dim=1)  # Average over all spatial locations

class IRFD(nn.Module):
    def __init__(self, max_resolution=256):
        super(IRFD, self).__init__()
        
        # Encoders (keeping ResNet50 backbones)
        self.Ei = self._create_encoder()  # Identity encoder
        self.Ee = self._create_encoder()  # Emotion encoder
        self.Ep = self._create_encoder()  # Pose encoder
        
        # CIPSGenerator-based generator
        self.Gd = CIPSGenerator(input_dim=2048*3,max_resolution=64)  # 2048*3 because we're concatenating 3 encoder outputs
        
        # StyleGAN Discriminator
        self.D = CIPSDiscriminator(input_dim=3, max_resolution=64)
        
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
        x_s_recon = self.Gd(gen_input_s, 64)
        x_t_recon = self.Gd(gen_input_t, 64)
        
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