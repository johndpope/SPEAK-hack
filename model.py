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
from memory_profiler import profile

IMAGE_SIZE = 512


        

class IRFD(nn.Module):
    def __init__(self,device):
        super(IRFD, self).__init__()
        
        # Encoders
        self.Ei = self._create_encoder()  # Identity encoder
        self.Ee = self._create_encoder()  # Emotion encoder
        self.Ep = self._create_encoder()  # Pose encoder
        
        # Generator
        input_dim = 3 * 2048  # Assuming each encoder outputs a 2048-dim feature
        self.Gd = IRFDGenerator512(input_dim=input_dim, ngf=64)
        

        self.Cm = nn.Linear(2048, 8) # 8 = num_emotion_classes
        self.to(device)
        
        
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
        
        # Randomly swap one type of feature
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
        x_s_recon = self.Gd(gen_input_s)
        x_t_recon = self.Gd(gen_input_t)
        
        # Apply softmax to emotion predictions
        emotion_pred_s = torch.softmax(self.Cm(fe_s.view(fe_s.size(0), -1)), dim=1)
        emotion_pred_t = torch.softmax(self.Cm(fe_t.view(fe_t.size(0), -1)), dim=1)
      
        
        return x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, emotion_pred_s, emotion_pred_t


class IRFDLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(IRFDLoss, self).__init__()
        self.alpha = alpha
        self.l2_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, x_s, x_t, x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, emotion_pred_s, emotion_pred_t, emotion_labels_s, emotion_labels_t):
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
        
        return total_loss, l_identity,l_cls, l_pose ,l_emotion, l_self





class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class IRFDGeneratorResBlocks(nn.Module):
    def __init__(self, input_dim, ngf=64):
        super(IRFDGeneratorResBlocks, self).__init__()
        
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(input_dim, ngf * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True)
        )
        
        self.resblocks = nn.Sequential(
            ResBlock(ngf * 32, ngf * 32),
            nn.Upsample(scale_factor=2),  # 8x8
            ResBlock(ngf * 32, ngf * 16),
            nn.Upsample(scale_factor=2),  # 16x16
            ResBlock(ngf * 16, ngf * 16),
            nn.Upsample(scale_factor=2),  # 32x32
            ResBlock(ngf * 16, ngf * 8),
            nn.Upsample(scale_factor=2),  # 64x64
            ResBlock(ngf * 8, ngf * 8),
            nn.Upsample(scale_factor=2),  # 128x128
            ResBlock(ngf * 8, ngf * 4),
            nn.Upsample(scale_factor=2),  # 256x256
            ResBlock(ngf * 4, ngf * 2),
            nn.Upsample(scale_factor=2),  # 512x512
            ResBlock(ngf * 2, ngf),
        )
        
        self.output = nn.Sequential(
            nn.Conv2d(ngf, 3, kernel_size=3, padding=1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.initial(x.view(x.size(0), -1, 1, 1))
        x = self.resblocks(x)
        return self.output(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        query = self.query(x).view(x.size(0), -1, x.size(2) * x.size(3)).permute(0, 2, 1)
        key = self.key(x).view(x.size(0), -1, x.size(2) * x.size(3))
        energy = torch.bmm(query, key)
        attention = torch.softmax(energy, dim=-1)
        value = self.value(x).view(x.size(0), -1, x.size(2) * x.size(3))
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(x.size(0), x.size(1), x.size(2), x.size(3))
        out = self.gamma * out + x
        return out

class IRFDGenerator512(nn.Module):
    def __init__(self, input_dim, ngf=64):
        super(IRFDGenerator512, self).__init__()
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_dim, ngf * 64, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 64, ngf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True),
            SelfAttention(ngf * 32),
            
            nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            SelfAttention(ngf * 8),
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
                
    def forward(self, x):
        return self.main(x.view(x.size(0), -1, 1, 1))
    


    
class CrossAttention(nn.Module):
    def __init__(self, latent_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads

        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.out = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, y):
        batch_size = x.size(0)

        # Compute query, key, and value vectors
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(y).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(y).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores and attended values
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        scores = torch.softmax(scores, dim=-1)
        attended = torch.matmul(scores, v).transpose(1, 2).contiguous().view(batch_size, -1, self.latent_dim)

        # Combine attended values with the original input
        out = self.out(attended)
        out = out + x

        return out
