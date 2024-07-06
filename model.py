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
from styleganv1 import StyleGenerator,StyleDiscriminator
import logging
import torchvision.utils as vutils
import matplotlib.pyplot as plt
class IRFD(nn.Module):
    def __init__(self, max_resolution=256):
        super(IRFD, self).__init__()
        
        # Encoders (keeping ResNet50 backbones)
        self.Ei = self._create_encoder()  # Identity encoder
        self.Ee = self._create_encoder()  # Emotion encoder
        self.Ep = self._create_encoder()  # Pose encoder
        

        self.Gd = StyleGenerator(input_dim=6144) 
        self.D = StyleDiscriminator()
        
        self.Cm = nn.Linear(2048, 8)  # 8 = num_emotion_classes
        
        self.max_resolution = max_resolution
        self.current_resolution = max_resolution

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def adjust_for_resolution(self,resolution):
        self.current_resolution = resolution


    def _create_encoder(self):
        encoder = resnet50(pretrained=True)
        return nn.Sequential(*list(encoder.children())[:-1])
    
    def _prepare_generator_input(self, *features):
        self.logger.debug(f"Preparing generator input. Feature shapes: {[f.shape for f in features]}")
        # Flatten and concatenate features
        concat_features = torch.cat([f.view(f.size(0), -1) for f in features], dim=1)
        self.logger.debug(f"Concatenated feature shape: {concat_features.shape}")
        return concat_features

    
    def _log_feature_stats(self, tensor, name):
        self.logger.debug(f"{name} stats: mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}")

    def _visualize_feature_maps(self, tensor, name, num_channels=8):
        if tensor.dim() == 4:
            grid = vutils.make_grid(tensor[:num_channels].unsqueeze(1), normalize=True, scale_each=True)
            vutils.save_image(grid, f"{name}_feature_maps.png")
    
    def forward(self, x_s, x_t):
        self.logger.debug(f"Input shapes: x_s={x_s.shape}, x_t={x_t.shape}")

        # Encode source and target images
        fi_s = checkpoint(self.Ei, x_s)
        fe_s = checkpoint(self.Ee, x_s)
        fp_s = checkpoint(self.Ep, x_s)
        
        fi_t = checkpoint(self.Ei, x_t)
        fe_t = checkpoint(self.Ee, x_t)
        fp_t = checkpoint(self.Ep, x_t)

        self.logger.debug(f"Encoded feature shapes: fi_s={fi_s.shape}, fe_s={fe_s.shape}, fp_s={fp_s.shape}")
        self._log_feature_stats(fi_s, "Identity features")
        self._log_feature_stats(fe_s, "Emotion features")
        self._log_feature_stats(fp_s, "Pose features")
        
        # Randomly swap one type of feature
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
        
        self.logger.debug(f"Generator input shapes: gen_input_s={gen_input_s.shape}, gen_input_t={gen_input_t.shape}")
        
        # Generate reconstructed images using StyleGANGenerator
        x_s_recon = self.Gd(gen_input_s)
        x_t_recon = self.Gd(gen_input_t)
        
        self.logger.debug(f"Reconstructed image shapes: x_s_recon={x_s_recon.shape}, x_t_recon={x_t_recon.shape}")
        self._visualize_feature_maps(x_s_recon, "source_reconstruction")
        self._visualize_feature_maps(x_t_recon, "target_reconstruction")
        
        # Apply softmax to emotion predictions
        emotion_pred_s = torch.softmax(self.Cm(fe_s.view(fe_s.size(0), -1)), dim=1)
        emotion_pred_t = torch.softmax(self.Cm(fe_t.view(fe_t.size(0), -1)), dim=1)
        
        self.logger.debug(f"Emotion prediction shapes: emotion_pred_s={emotion_pred_s.shape}, emotion_pred_t={emotion_pred_t.shape}")
      
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

    def check_tensor(self,tensor, name):
        if torch.isnan(tensor).any():
            print(f"Warning: NaN values found in {name}")
        if torch.isinf(tensor).any():
            print(f"Warning: Inf values found in {name}")


    def detect_landmarks(self, images):
        batch_size = images.size(0)
        landmarks_batch = []

        for i in range(batch_size):
            image = images[i]
            # Before the conversion
            self.check_tensor(image, "image tensor")

            image_np = image.detach().cpu().permute(1, 2, 0).numpy()
            image_np = np.clip(image_np, 0, 1)  # Ensure values are between 0 and 1
            image_np = (image_np * 255).astype(np.uint8)
            
            
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