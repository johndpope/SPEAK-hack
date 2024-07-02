import torch
import torch.nn as nn
from torchvision.models import resnet50
import colored_traceback.auto
import torch.nn.functional as F
from torchvision import models
import lpips
from torch.utils.checkpoint import checkpoint

import mediapipe as mp
import numpy as np

def landmark_alignment_loss(real_landmarks, generated_landmarks):
    # Calculate Euclidean distance between corresponding landmarks
    distances = np.linalg.norm(real_landmarks - generated_landmarks, axis=1)
    # Return mean distance as the loss
    return np.mean(distances)



class CustomResNet50(torch.nn.Module):
    def __init__(self, fine_tune_from=0):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = torch.nn.Sequential(
            *[layer for layer in resnet.children()][:-2]
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        
        # Freeze layers before fine_tune_from
        for i, (name, param) in enumerate(self.features.named_parameters()):
            if i < fine_tune_from:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x.squeeze(-1).squeeze(-1)
    

class SimpleGenerator(nn.Module):
    def __init__(self, input_dim, ngf=64, max_resolution=256):
        super(SimpleGenerator, self).__init__()
        
        self.input_dim = input_dim
        self.ngf = ngf
        self.max_resolution = max_resolution
        
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(input_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
        )
        
        self.layers = nn.ModuleList()
        current_ngf = ngf * 8
        current_size = 4
        while current_size < max_resolution:
            next_ngf = current_ngf // 2
            self.layers.append(nn.Sequential(
                nn.ConvTranspose2d(current_ngf, next_ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(next_ngf),
                nn.ReLU(True)
            ))
            current_ngf = next_ngf
            current_size *= 2
        
        self.final = None
    

    def forward(self, x, target_resolution):
        x = self.initial(x.view(x.size(0), -1, 1, 1))
        if torch.isnan(x).any():
            print("NaN detected after initial layer in SimpleGenerator")
            return None

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if torch.isnan(x).any():
                print(f"NaN detected after layer {i} in SimpleGenerator")
                return None
            if x.size(-1) >= target_resolution:
                break

        if x.size(-1) != target_resolution:
            x = F.interpolate(x, size=(target_resolution, target_resolution), mode='bilinear', align_corners=False)

        if self.final is None or self.final[0].in_channels != x.size(1):
            self.final = nn.Sequential(
                nn.Conv2d(x.size(1), 3, 3, 1, 1),
                nn.Tanh()
            ).to(x.device)

        output = self.final(x)
        if torch.isnan(output).any():
            print("NaN detected in final output of SimpleGenerator")
            return None

        # Ensure output is in [-1, 1] range
        output = torch.tanh(output)
        return output

class BasicGenerator64(nn.Module):
    def __init__(self, input_dim=2048*3):
        super(BasicGenerator64, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64 * 64 * 3),  # Output for 64x64 RGB image
            nn.Tanh()  # Normalize output to [-1, 1]
        )

    def forward(self, x,bla):
        x = self.main(x)
        return x.view(-1, 3, 64, 64)  # Reshape to image dimensions


       
class BasicGenerator256(nn.Module):
    def __init__(self, input_dim=2048*3):
        super(BasicGenerator256, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * 256 * 3),  
            nn.Tanh()  # Normalize output to [-1, 1]
        )

    def forward(self, x,bla):
        x = self.main(x)
        return x.view(-1, 3, 256, 256)  # Reshape to image dimensions

import math

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
    def __init__(self):
        super(IRFD, self).__init__()
        self.Ei = CustomResNet50()  # Identity encoder
        self.Ee = CustomResNet50()  # Emotion encoder
        self.Ep = CustomResNet50()  # Pose encoder
        self.Gd = SimpleGenerator(input_dim=2048*3)  # Assuming SimpleGenerator is defined elsewhere

    def forward(self, x_s, x_t):
        # Identity forward pass
        fi_s = self.Ei(x_s)
        fi_t = self.Ei(x_t)

        # Emotion forward pass
        fe_s = self.Ee(x_s)
        fe_t = self.Ee(x_t)

        # Pose forward pass
        fp_s = self.Ep(x_s)
        fp_t = self.Ep(x_t)

        # Combine features
        gen_input_s = torch.cat([fi_s, fe_s, fp_s], dim=1)
        gen_input_t = torch.cat([fi_t, fe_t, fp_t], dim=1)

        # Generate reconstructed images
        x_s_recon = self.Gd(gen_input_s, x_s.shape[2])
        x_t_recon = self.Gd(gen_input_t, x_t.shape[2])

        return x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t

    def forward_identity(self, x):
        return self.Ei(x)

    def forward_emotion(self, x):
        return self.Ee(x)

    def forward_pose(self, x):
        return self.Ep(x)


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
        l_identity = self.identity_loss(fi_s, fi_t)
        l_emotion = self.emotion_loss(fe_s, fe_t, emotion_labels_s, emotion_labels_t)
        l_pose = self.pose_loss(fp_s, fp_t)
        l_landmark = self.landmark_loss(x_s, x_s_recon) + self.landmark_loss(x_t, x_t_recon)
        l_recon = self.reconstruction_loss(x_s, x_t, x_s_recon, x_t_recon)

        return l_pose + l_landmark, l_emotion, l_identity, l_recon

    def __del__(self):
        # Clean up MediaPipe resources
        self.face_mesh.close()