import torch
import torch.nn as nn
from torchvision.models import resnet50
import colored_traceback.auto
import torch.nn.functional as F
from torchvision import models
import lpips


FEATURE_SIZE_AVG_POOL = 2 # use 2 - not 4. https://github.com/johndpope/MegaPortrait-hack/issues/23
FEATURE_SIZE = (2, 2) 


class CustomResNet50(torch.nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = torch.nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

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


class IRFD(nn.Module):
    def __init__(self, input_dim=2048, ngf=64, max_resolution=256):
        super(IRFD, self).__init__()
        
        self.enable_swapping = True
        # Encoders
        self.Ei = CustomResNet50()  # Identity encoder
        self.Ee = CustomResNet50()  # Emotion encoder
        self.Ep = CustomResNet50()  # Pose encoder
        
        # Generator
        self.Gd = SimpleGenerator(input_dim=input_dim * 3, ngf=ngf, max_resolution=max_resolution)

        self.Cm = nn.Linear(2048, 8) # 8 = num_emotion_classes

    
    def forward(self, x_s, x_t):
        # Check input range
        # Clamp input values to [-1, 1] range
        x_s = torch.clamp(x_s, -1, 1)
        x_t = torch.clamp(x_t, -1, 1)

        if x_s.min() < -1 or x_s.max() > 1 or x_t.min() < -1 or x_t.max() > 1:
            print(f"Input range warning: x_s min={x_s.min():.2f}, max={x_s.max():.2f}, x_t min={x_t.min():.2f}, max={x_t.max():.2f}")
    
        # print(f"Input shapes - x_s: {x_s.shape}, x_t: {x_t.shape}")
        # print(f"Input min/max/mean - x_s: {x_s.min():.4f}/{x_s.max():.4f}/{x_s.mean():.4f}, x_t: {x_t.min():.4f}/{x_t.max():.4f}/{x_t.mean():.4f}")
        
        # Encode source and target images
        fi_s = self.Ei(x_s)
        fe_s = self.Ee(x_s)
        fp_s = self.Ep(x_s)
        
        fi_t = self.Ei(x_t)
        fe_t = self.Ee(x_t)
        fp_t = self.Ep(x_t)
        
        print(f"fi_s shape: {fi_s.shape}")
        print(f"fi_s statistics:")
        print(f"  Min: {fi_s.min().item():.4f}")
        print(f"  Max: {fi_s.max().item():.4f}")
        print(f"  Mean: {fi_s.mean().item():.4f}")
        print(f"  Std: {fi_s.std().item():.4f}")

        print(f"fi_t shape: {fi_t.shape}")
        print(f"fi_t statistics:")
        print(f"  Min: {fi_t.min().item():.4f}")
        print(f"  Max: {fi_t.max().item():.4f}")
        print(f"  Mean: {fi_t.mean().item():.4f}")
        print(f"  Std: {fi_t.std().item():.4f}")


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
        
        print("Feature extraction successful!")
        print(f"Feature shape: {x_s_recon.shape}")
        print(f"Feature statistics:")
        print(f"  Min: {x_s_recon.min().item():.4f}")
        print(f"  Max: {x_s_recon.max().item():.4f}")
        print(f"  Mean: {x_s_recon.mean().item():.4f}")
        print(f"  Std: {x_s_recon.std().item():.4f}")

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



class IRFDLoss(nn.Module):
    def __init__(self, alpha=0.1, lpips_weight=.2):
        super(IRFDLoss, self).__init__()
        self.alpha = alpha
        self.lpips_weight = lpips_weight
        self.l2_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.lpips_loss = lpips.LPIPS(net='vgg')  # You can also use 'vgg' instead of 'alex'
    
    def forward(self, x_s, x_t, x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, emotion_pred_s, emotion_pred_t, emotion_labels_s, emotion_labels_t):
        # Assertions to check for NaNs
        assert not torch.isnan(x_s).any(), "NaN detected in x_s"
        assert not torch.isnan(x_t).any(), "NaN detected in x_t"
        assert not torch.isnan(x_s_recon).any(), "NaN detected in x_s_recon"
        assert not torch.isnan(x_t_recon).any(), "NaN detected in x_t_recon"
        assert not torch.isnan(fi_s).any(), "NaN detected in fi_s"
        assert not torch.isnan(fe_s).any(), "NaN detected in fe_s"
        assert not torch.isnan(fp_s).any(), "NaN detected in fp_s"
        assert not torch.isnan(fi_t).any(), "NaN detected in fi_t"
        assert not torch.isnan(fe_t).any(), "NaN detected in fe_t"
        assert not torch.isnan(fp_t).any(), "NaN detected in fp_t"
        assert not torch.isnan(emotion_pred_s).any(), "NaN detected in emotion_pred_s"
        assert not torch.isnan(emotion_pred_t).any(), "NaN detected in emotion_pred_t"
        assert not torch.isnan(emotion_labels_s).any(), "NaN detected in emotion_labels_s"
        assert not torch.isnan(emotion_labels_t).any(), "NaN detected in emotion_labels_t"



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


