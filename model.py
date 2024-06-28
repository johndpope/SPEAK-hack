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

IMAGE_SIZE = 512


        

class IRFD(nn.Module):
    def __init__(self):
        super(IRFD, self).__init__()
        
        # Encoders
        self.Ei = self._create_encoder()  # Identity encoder
        self.Ee = self._create_encoder()  # Emotion encoder
        self.Ep = self._create_encoder()  # Pose encoder
        
        # Generator
        input_dim = 3 * 2048  # Assuming each encoder outputs a 2048-dim feature
        # self.Gd = IRFDGeneratorSPADE(input_dim=input_dim, ngf=64, label_nc=input_dim)
        self.Gd = IRFDGeneratorResBlocks(input_dim=input_dim, ngf=64)


        self.Cm = nn.Linear(2048, 8) # 8 = num_emotion_classes

        # Initialize the Emotion Recognizer
        model_name = 'enet_b0_8_va_mtl'  # Adjust as needed depending on the model availability
        self.fer = HSEmotionRecognizer(model_name=model_name)
        self.emotion_idx_to_class = {0: 'angry', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 
                                     5: 'neutral', 6: 'sad', 7: 'surprise'}
        
        
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
        
        # Generate reconstructed images
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
        # print("x_s:",x_s.shape)
        # print("x_s_recon:",x_s_recon.shape)
        # print("x_t:",x_t.shape)
        # print("x_t_recon:",x_t_recon.shape)
        
        l_self = self.l2_loss(x_s, x_s_recon) + self.l2_loss(x_t, x_t_recon)
        
        # Total loss
        total_loss = l_identity + l_cls + l_pose + l_emotion + l_self
        
        return total_loss, l_identity,l_cls, l_pose ,l_emotion, l_self





class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, label_nc):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)

    def forward(self, x, segmap):
        x_s = self.shortcut(x, segmap)
        dx = self.conv_0(self.actvn(self.norm_0(x, segmap)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, segmap)))
        out = x_s + dx
        return out

    def shortcut(self, x, segmap):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, segmap))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

# 512
class IRFDGeneratorSPADE(nn.Module):
    def __init__(self, input_dim, ngf=64, label_nc=3*2048):  # Assuming concatenated features
        super(IRFDGeneratorSPADE, self).__init__()
        
        self.fc = nn.Linear(input_dim, 16 * ngf * 4 * 4)
        
        self.head_0 = SPADEResnetBlock(16 * ngf, 16 * ngf, label_nc)
        self.G_middle_0 = SPADEResnetBlock(16 * ngf, 16 * ngf, label_nc)
        self.G_middle_1 = SPADEResnetBlock(16 * ngf, 16 * ngf, label_nc)
        
        self.up_0 = SPADEResnetBlock(16 * ngf, 8 * ngf, label_nc)
        self.up_1 = SPADEResnetBlock(8 * ngf, 4 * ngf, label_nc)
        self.up_2 = SPADEResnetBlock(4 * ngf, 2 * ngf, label_nc)
        self.up_3 = SPADEResnetBlock(2 * ngf, ngf, label_nc)
        self.up_4 = SPADEResnetBlock(ngf, ngf // 2, label_nc)
        self.up_5 = SPADEResnetBlock(ngf // 2, ngf // 4, label_nc)
        
        self.conv_img = nn.Conv2d(ngf // 4, 3, 3, padding=1)
        
    def forward(self, input):
        segmap = input.view(input.size(0), -1, 1, 1).expand(-1, -1, 4, 4)
        x = self.fc(input.view(input.size(0), -1))
        x = x.view(-1, 16 * 64, 4, 4)
        
        x = self.head_0(x, segmap)
        x = F.interpolate(x, scale_factor=2)  # 8x8
        x = self.G_middle_0(x, segmap)
        x = self.G_middle_1(x, segmap)
        
        x = F.interpolate(x, scale_factor=2)  # 16x16
        x = self.up_0(x, segmap)
        x = F.interpolate(x, scale_factor=2)  # 32x32
        x = self.up_1(x, segmap)
        x = F.interpolate(x, scale_factor=2)  # 64x64
        x = self.up_2(x, segmap)
        x = F.interpolate(x, scale_factor=2)  # 128x128
        x = self.up_3(x, segmap)
        x = F.interpolate(x, scale_factor=2)  # 256x256
        x = self.up_4(x, segmap)
        x = F.interpolate(x, scale_factor=2)  # 512x512
        x = self.up_5(x, segmap)
        x = F.interpolate(x, scale_factor=2)  # 1024x1024
        
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)
        
        return x



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




class IRFDGenerator512(nn.Module):
    def __init__(self, input_dim, ngf=64):
        super(IRFDGenerator512, self).__init__()
        
        self.main = nn.Sequential(
            # Input is the concatenated identity, emotion and pose embeddings
            # input_dim = 3 * 2048 = 6144 (assuming ResNet-50 encoders)
            nn.ConvTranspose2d(input_dim, ngf * 64, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 64, ngf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
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
            
            # Output is a reconstructed image of shape (3, 512, 512)
        )
        
        # Initialization scheme
        # This specific initialization scheme (normal distribution with the chosen mean and standard deviation) is based on the recommendations from the DCGAN paper (Radford et al., 2016), which has been found to work well for various GAN architectures.
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
                
    def forward(self, x):
        return self.main(x.view(x.size(0), -1, 1, 1))


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

class IRFDGenerator(nn.Module):
    def __init__(self, input_dim, ngf=64):
        super(IRFDGenerator, self).__init__()
        
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
            nn.Upsample(scale_factor=2),  # 1024x1024
        )
        
        self.output = nn.Sequential(
            nn.Conv2d(ngf, 3, kernel_size=3, padding=1, bias=False),
            nn.Tanh()        )
        
    def forward(self, x):
        x = self.initial(x.view(x.size(0), -1, 1, 1))
        x = self.resblocks(x)
        return self.output(x)



class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    
    def forward(self, audio):
        return self.wav2vec(audio).last_hidden_state

class EditingModule(nn.Module):
    def __init__(self):
        super(EditingModule, self).__init__()
        # Implement the editing module as described in the paper
        # This should combine facial features with audio features
    
    def forward(self, facial_features, audio_features):
        # Combine and process features
        pass

class TalkingHeadGenerator(nn.Module):
    def __init__(self):
        super(TalkingHeadGenerator, self).__init__()
        # Implement the global generator Gg as described in the paper
    
    def forward(self, edited_features):
        # Generate the final talking head video
        pass

class SPEAK(nn.Module):
    def __init__(self):
        super(SPEAK, self).__init__()
        self.irfd = IRFD()
        self.audio_encoder = AudioEncoder()
        self.editing_module = EditingModule()
        self.talking_head_generator = TalkingHeadGenerator()
    
    def forward(self, identity_image, emotion_video, pose_video, audio):
        # Implement the full SPEAK pipeline
        pass

# Additional loss functions
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Implement perceptual loss using a pre-trained VGG network

class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        # Implement GAN loss