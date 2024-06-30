import torch
import torch.nn as nn
from torchvision.models import resnet50
import colored_traceback.auto
import torch.nn.functional as F


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
    def __init__(self, input_dim, ngf=64, max_resolution=512):
        super(IRFDGeneratorResBlocks, self).__init__()
        
        self.input_dim = input_dim
        self.ngf = ngf
        self.max_resolution = max_resolution
        
        # Initial upsampling
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(input_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
        )
        
        # Upsampling layers
        self.layers = nn.ModuleList()
        current_ngf = ngf * 8
        current_resolution = 4
        while current_resolution < max_resolution:
            next_ngf = current_ngf // 2
            self.layers.append(nn.Sequential(
                ResBlock(current_ngf, next_ngf),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ))
            current_ngf = next_ngf
            current_resolution *= 2
        
        # We'll create the final conv layer in the forward pass
        self.tanh = nn.Tanh()
    
    def forward(self, x, target_resolution):
        # print(f"Generator input shape: {x.shape}")
        
        # Initial upsampling
        x = self.initial(x.view(x.size(0), -1, 1, 1))
        # print(f"After initial upsampling: {x.shape}")
        
        # Apply upsampling layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # print(f"After layer {i+1}: {x.shape}")
            if x.size(-1) >= target_resolution:
                break
        
        # Ensure correct output size
        if x.size(-1) != target_resolution:
            x = F.interpolate(x, size=(target_resolution, target_resolution), mode='bilinear', align_corners=False)
        
        # print(f"Before final conv: {x.shape}")
        
        # Create and apply the final convolution layer
        final_conv = nn.Conv2d(x.size(1), 3, kernel_size=3, padding=1, bias=False).to(x.device)
        x = final_conv(x)
        x = self.tanh(x)
        
        # print(f"Generator output shape: {x.shape}")
        
        return x

    def __repr__(self):
        return f"IRFDGeneratorResBlocks(input_dim={self.input_dim}, ngf={self.ngf}, max_resolution={self.max_resolution})"

        
class IRFD(nn.Module):
    def __init__(self, input_dim=2048, ngf=64, max_resolution=512):
        super(IRFD, self).__init__()
        
        # Encoders
        self.Ei = self._create_encoder()  # Identity encoder
        self.Ee = self._create_encoder()  # Emotion encoder
        self.Ep = self._create_encoder()  # Pose encoder
        
        # Generator
        self.Gd = IRFDGeneratorResBlocks(input_dim=input_dim * 3, ngf=ngf, max_resolution=max_resolution)

        self.Cm = nn.Linear(2048, 8) # 8 = num_emotion_classes

    def _create_encoder(self):
        encoder = resnet50(pretrained=True)
        return nn.Sequential(*list(encoder.children())[:-1])
    
    def forward(self, x_s, x_t):
        # print(f"Input shapes - x_s: {x_s.shape}, x_t: {x_t.shape}")
        
        # Encode source and target images
        fi_s = self.Ei(x_s)
        fe_s = self.Ee(x_s)
        fp_s = self.Ep(x_s)
        
        fi_t = self.Ei(x_t)
        fe_t = self.Ee(x_t)
        fp_t = self.Ep(x_t)
        
        # print(f"Encoder outputs - fi_s: {fi_s.shape}, fe_s: {fe_s.shape}, fp_s: {fp_s.shape}")
        
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
        
        # print(f"Generator input - gen_input_s: {gen_input_s.shape}")
        
        # Generate reconstructed images
        target_resolution = x_s.shape[2]  # Assuming square images
        x_s_recon = self.Gd(gen_input_s, target_resolution)
        x_t_recon = self.Gd(gen_input_t, target_resolution)
        
        # print(f"Reconstructed images - x_s_recon: {x_s_recon.shape}, x_t_recon: {x_t_recon.shape}")
        
        # Apply softmax to emotion predictions
        emotion_pred_s = torch.softmax(self.Cm(fe_s.view(fe_s.size(0), -1)), dim=1)
        emotion_pred_t = torch.softmax(self.Cm(fe_t.view(fe_t.size(0), -1)), dim=1)
      
        return x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, emotion_pred_s, emotion_pred_t

    def __repr__(self):
        return "IRFD(input_dim=2048, ngf=64, max_resolution=512)"

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


