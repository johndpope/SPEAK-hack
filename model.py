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

IMAGE_SIZE = 512

class IRFDGenerator(nn.Module):
        def __init__(self, input_dim, ngf=64):
            super(IRFDGenerator, self).__init__()
            
            self.main = nn.Sequential(
                # Input is the concatenated identity, emotion and pose embeddings
                # input_dim = 3 * 2048 = 6144 (assuming ResNet-50 encoders)
                nn.ConvTranspose2d(input_dim, ngf * 8, 4, 1, 0, bias=False),
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
                
                # Output is a reconstructed image of shape (3, IMAGE_SIZE, IMAGE_SIZE)
            )
            
            # This specific initialization scheme (normal distribution with the chosen mean and standard deviation) is based on the recommendations from the DCGAN paper (Radford et al., 2016), which has been found to work well for various GAN architectures.
            for m in self.modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight.data, 0.0, 0.02)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0)
                    
        def forward(self, x):
            return self.main(x.view(x.size(0), -1, 1, 1))
        

class IRFD(nn.Module):
    def __init__(self):
        super(IRFD, self).__init__()
        
        # Encoders
        self.Ei = self._create_encoder()  # Identity encoder
        self.Ee = self._create_encoder()  # Emotion encoder
        self.Ep = self._create_encoder()  # Pose encoder
        
        # Generator
        self.Gd =  IRFDGenerator(input_dim = 3 * 2048) 
        
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
        
        # Generate reconstructed images
        print("fi_s:",fi_s.shape)
        print("fe_s:",fe_s.shape)
        print("fp_s:",fp_s.shape)
        
        x_s_recon = self.Gd(torch.cat([fi_s, fe_s, fp_s], dim=1))
        x_t_recon = self.Gd(torch.cat([fi_t, fe_t, fp_t], dim=1))
        
        return x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t

class IRFDLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(IRFDLoss, self).__init__()
        self.alpha = alpha
        self.l2_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, x_s, x_t, x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, emotion_labels_s, emotion_labels_t):
        # Identity loss
        l_identity = torch.max(
            self.l2_loss(fi_s, fi_t) - self.l2_loss(fi_s, fi_s) + self.alpha,
            torch.tensor(0.0).to(fi_s.device)
        )
        
        # Classification loss
        l_cls = self.ce_loss(fe_s, emotion_labels_s) + self.ce_loss(fe_t, emotion_labels_t)
        
        # Pose loss
        l_pose = self.l2_loss(fp_s, fp_t)
        
        # Emotion loss
        l_emotion = self.l2_loss(fe_s, fe_t)
        
        # Self-reconstruction loss
        l_self = self.l2_loss(x_s, x_s_recon) + self.l2_loss(x_t, x_t_recon)
        
        # Total loss
        total_loss = l_identity + l_cls + l_pose + l_emotion + l_self
        
        return total_loss