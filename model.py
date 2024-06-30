import torch
import torch.nn as nn
from torchvision.models import resnet50
import colored_traceback.auto
import torch.nn.functional as F
from torchvision import models
import lpips


class SimpleGenerator(nn.Module):
    def __init__(self, input_dim, ngf=64, max_resolution=256):
        super(SimpleGenerator, self).__init__()
        
        self.input_dim = input_dim
        self.ngf = ngf
        self.max_resolution = max_resolution
        
        # print(f"Initializing SimpleGenerator with input_dim={input_dim}, ngf={ngf}, max_resolution={max_resolution}")
        
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
            # print(f"Added layer: input_channels={current_ngf*2}, output_channels={current_ngf}, size={current_size}")
        
        self.final = None  # We'll create this in the forward pass
    
    def forward(self, x, target_resolution):
        # print(f"\nSimpleGenerator forward pass:")
        # print(f"Input shape: {x.shape}, target_resolution: {target_resolution}")
        
        x = self.initial(x.view(x.size(0), -1, 1, 1))
        # print(f"After initial layer: {x.shape}")
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # print(f"After layer {i+1}: {x.shape}")
            if x.size(-1) >= target_resolution:
                # print(f"Reached or exceeded target resolution. Stopping at layer {i+1}")
                break
        
        # Ensure correct output size
        if x.size(-1) != target_resolution:
            # print(f"Resizing from {x.shape[2:]} to {target_resolution}")
            x = F.interpolate(x, size=(target_resolution, target_resolution), mode='bilinear', align_corners=False)
            # print(f"After resizing: {x.shape}")
        
        # Create the final layer dynamically based on the current number of channels
        if self.final is None or self.final[0].in_channels != x.size(1):
            # print(f"Creating new final layer. Input channels: {x.size(1)}")
            self.final = nn.Sequential(
                nn.Conv2d(x.size(1), 3, 3, 1, 1),
                nn.Tanh()
            ).to(x.device)
        # else:
            # print(f"Using existing final layer. Input channels: {self.final[0].in_channels}")
        
        output = self.final(x)
        # print(f"Final output shape: {output.shape}")
        
        return output

    def __repr__(self):
        return f"SimpleGenerator(input_dim={self.input_dim}, ngf={self.ngf}, max_resolution={self.max_resolution})"




class IRFD(nn.Module):
    def __init__(self, input_dim=2048, ngf=64, max_resolution=256):
        super(IRFD, self).__init__()
        
        self.enable_swapping = True
        # Encoders
        self.Ei = self._create_encoder()  # Identity encoder
        self.Ee = self._create_encoder()  # Emotion encoder
        self.Ep = self._create_encoder()  # Pose encoder
        
        # Generator
        self.Gd = SimpleGenerator(input_dim=input_dim * 3, ngf=ngf, max_resolution=max_resolution)

        self.Cm = nn.Linear(2048, 8) # 8 = num_emotion_classes

    def _create_encoder(self):
        encoder = resnet50(pretrained=True)
        return nn.Sequential(*list(encoder.children())[:-1])
    
    def forward(self, x_s, x_t):
        # print(f"Input shapes - x_s: {x_s.shape}, x_t: {x_t.shape}")
        # print(f"Input min/max/mean - x_s: {x_s.min():.4f}/{x_s.max():.4f}/{x_s.mean():.4f}, x_t: {x_t.min():.4f}/{x_t.max():.4f}/{x_t.mean():.4f}")
        
        # Encode source and target images
        fi_s = self.Ei(x_s)
        fe_s = self.Ee(x_s)
        fp_s = self.Ep(x_s)
        
        fi_t = self.Ei(x_t)
        fe_t = self.Ee(x_t)
        fp_t = self.Ep(x_t)
        
        
         # Conditional swapping
        if self.enable_swapping:
            swap_type = torch.randint(0, 3, (1,)).item()
            if swap_type == 0:
                fi_s, fi_t = fi_t, fi_s
                # print("Swapped identity features")
            elif swap_type == 1:
                fe_s, fe_t = fe_t, fe_s
                # print("Swapped emotion features")
            else:
                fp_s, fp_t = fp_t, fp_s
                # print("Swapped pose features")
        # else:
            # print("Swapping disabled")

        # Concatenate features for generator input
        gen_input_s = torch.cat([fi_s, fe_s, fp_s], dim=1).squeeze(-1).squeeze(-1)
        gen_input_t = torch.cat([fi_t, fe_t, fp_t], dim=1).squeeze(-1).squeeze(-1)
        
        # print(f"Generator input - gen_input_s: {gen_input_s.shape}")
        # print(f"Generator input min/max/mean - gen_input_s: {gen_input_s.min():.4f}/{gen_input_s.max():.4f}/{gen_input_s.mean():.4f}")
        
        # Generate reconstructed images
        target_resolution = x_s.shape[2]  # Assuming square images
        x_s_recon = self.Gd(gen_input_s, target_resolution)
        x_t_recon = self.Gd(gen_input_t, target_resolution)
        
        # print(f"Reconstructed images - x_s_recon: {x_s_recon.shape}, x_t_recon: {x_t_recon.shape}")
        # print(f"Reconstructed min/max/mean - x_s_recon: {x_s_recon.min():.4f}/{x_s_recon.max():.4f}/{x_s_recon.mean():.4f}")
        
        # Apply softmax to emotion predictions
        emotion_pred_s = torch.softmax(self.Cm(fe_s.view(fe_s.size(0), -1)), dim=1)
        emotion_pred_t = torch.softmax(self.Cm(fe_t.view(fe_t.size(0), -1)), dim=1)
        
        # print(f"Emotion predictions - emotion_pred_s: {emotion_pred_s.shape}")
        # print(f"Emotion pred min/max/mean - emotion_pred_s: {emotion_pred_s.min():.4f}/{emotion_pred_s.max():.4f}/{emotion_pred_s.mean():.4f}")
      
        return x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, emotion_pred_s, emotion_pred_t

    def __repr__(self):
        return "IRFD(input_dim=2048, ngf=64, max_resolution=256)"

# class VGGFacePerceptualLoss(nn.Module):
#     def __init__(self):
#         super(VGGFacePerceptualLoss, self).__init__()
#         # Load a pre-trained VGG model (you might want to use a VGG model specifically trained on faces if available)
#         vgg = models.vgg16(pretrained=True)
#         # Use the first few layers for perceptual loss
#         self.feature_extractor = nn.Sequential(*list(vgg.features)[:16])
#         # Freeze the VGG parameters
#         for param in self.feature_extractor.parameters():
#             param.requires_grad = False

#     def forward(self, x, y):
#         x_features = self.feature_extractor(x)
#         y_features = self.feature_extractor(y)
#         return F.mse_loss(x_features, y_features)

# class IRFDLoss(nn.Module):
#     def __init__(self, alpha=0.1, vgg_weight=0.1):
#         super(IRFDLoss, self).__init__()
#         self.alpha = alpha
#         self.vgg_weight = vgg_weight
#         self.l2_loss = nn.MSELoss()
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.vgg_loss = VGGFacePerceptualLoss()
    
#     def forward(self, x_s, x_t, x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, emotion_pred_s, emotion_pred_t, emotion_labels_s, emotion_labels_t):
#         # Ensure all images have the same size
#         x_s = F.interpolate(x_s, size=x_s_recon.shape[2:], mode='bilinear', align_corners=False)
#         x_t = F.interpolate(x_t, size=x_t_recon.shape[2:], mode='bilinear', align_corners=False)

#         # Ensure all inputs have the same batch size
#         batch_size = x_s.size(0)
        
#         # Reshape emotion predictions and labels if necessary
#         emotion_pred_s = emotion_pred_s.view(batch_size, -1)
#         emotion_pred_t = emotion_pred_t.view(batch_size, -1)
#         emotion_labels_s = emotion_labels_s.view(batch_size)
#         emotion_labels_t = emotion_labels_t.view(batch_size)

#         # Identity loss
#         l_identity = torch.max(
#             self.l2_loss(fi_s, fi_t) - self.l2_loss(fi_s, fi_s) + self.alpha,
#             torch.tensor(0.0).to(fi_s.device)
#         )
        
#         # Classification loss
#         l_cls = self.ce_loss(emotion_pred_s, emotion_labels_s) + self.ce_loss(emotion_pred_t, emotion_labels_t)
        
#         # Pose loss
#         l_pose = self.l2_loss(fp_s, fp_t)
        
#         # Emotion loss
#         l_emotion = self.l2_loss(fe_s, fe_t)
        
#         # Self-reconstruction loss
#         l_self = self.l2_loss(x_s, x_s_recon) + self.l2_loss(x_t, x_t_recon)
        
#         # VGG Perceptual loss
#         l_vgg = self.vgg_loss(x_s, x_s_recon) + self.vgg_loss(x_t, x_t_recon)
        
#         # Total loss
#         total_loss = l_identity + l_cls + l_pose + l_emotion + l_self + self.vgg_weight * l_vgg
        
#         return total_loss, l_identity, l_cls, l_pose, l_emotion, l_self, l_vgg



class IRFDLoss(nn.Module):
    def __init__(self, alpha=0.1, lpips_weight=1):
        super(IRFDLoss, self).__init__()
        self.alpha = alpha
        self.lpips_weight = lpips_weight
        self.l2_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.lpips_loss = lpips.LPIPS(net='vgg')  # You can also use 'vgg' instead of 'alex'
    
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
        
        # LPIPS Perceptual loss
        l_lpips = self.lpips_loss(x_s, x_s_recon).mean() + self.lpips_loss(x_t, x_t_recon).mean()
        
        # Total loss
        total_loss = l_identity + l_cls + l_pose + l_emotion + l_self + self.lpips_weight * l_lpips
        print("total_loss:",total_loss)
        return total_loss, l_identity, l_cls, l_pose, l_emotion, l_self, l_lpips