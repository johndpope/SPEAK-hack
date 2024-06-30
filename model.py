import torch
import torch.nn as nn
from torchvision.models import resnet50
import colored_traceback.auto
import torch.nn.functional as F
from torchvision import models
import lpips


FEATURE_SIZE_AVG_POOL = 2 # use 2 - not 4. https://github.com/johndpope/MegaPortrait-hack/issues/23
FEATURE_SIZE = (2, 2) 


# we need custom resnet blocks - so use the ResNet50  es.shape: torch.Size([1, 512, 1, 1])
# n.b. emoportraits reduced this from 512 -> 128 dim - these are feature maps / identity fingerprint of image 
class CustomResNet50(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        resnet = models.resnet50(*args, **kwargs)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
      #  self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        # Remove the last residual block (layer4)
        # self.layer4 = resnet.layer4
        
        # Add an adaptive average pooling layer
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(FEATURE_SIZE_AVG_POOL)
        
        # Add a 1x1 convolutional layer to reduce the number of channels to 512
        self.conv_reduce = nn.Conv2d(1024, 512, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # Remove the forward pass through layer4
        # x = self.layer4(x)
        
        # Apply adaptive average pooling
        x = self.adaptive_avg_pool(x)
        
        # Apply the 1x1 convolutional layer to reduce the number of channels
        x = self.conv_reduce(x)
        
        return x


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
        assert not torch.isnan(x).any(), "NaN detected after initial layer"

        for i, layer in enumerate(self.layers):
            x = layer(x)
            assert not torch.isnan(x).any(), f"NaN detected after layer {i+1}"
            if x.size(-1) >= target_resolution:
                break
        
        if x.size(-1) != target_resolution:
            x = F.interpolate(x, size=(target_resolution, target_resolution), mode='bilinear', align_corners=False)
            assert not torch.isnan(x).any(), "NaN detected after resizing"

        if self.final is None or self.final[0].in_channels != x.size(1):
            self.final = nn.Sequential(
                nn.Conv2d(x.size(1), 3, 3, 1, 1),
                nn.Tanh()
            ).to(x.device)
        
        output = self.final(x)
        assert not torch.isnan(output).any(), "NaN detected in final output"
        
        return output

    def __repr__(self):
        return f"SimpleGenerator(input_dim={self.input_dim}, ngf={self.ngf}, max_resolution={self.max_resolution})"




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