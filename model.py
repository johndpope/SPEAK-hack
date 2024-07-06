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
import torch
import torch.nn as nn
import torch.nn.functional as F
from mysixdrepnet import  SixDRepNet_Detector
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
import dlib
import numpy as np
import logging
import cv2

class IRFDLoss(nn.Module):
    def __init__(self, config, device):
        super(IRFDLoss, self).__init__()
        self.device = device
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Load pre-trained models
        self.rotation_net =  SixDRepNet_Detector()

        self.emotion_model = HSEmotionRecognizer(model_name='enet_b0_8_va_mtl')
        
        # Initialize dlib's face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Loss weights
        self.face_recognition_weight = config['weights']['face_recognition']
        self.emotion_weight = config['weights']['emotion']
        self.landmark_weight = config['weights']['landmark']
        
        # Other loss functions
        self.l2_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def tensor_to_numpy(self, tensor):
        return tensor.detach().cpu().numpy()

    def get_landmarks(self, image):
        try:
            np_image = self.tensor_to_numpy(image)
            np_image = np.transpose(np_image, (1, 2, 0))
            np_image = (np_image * 255).astype(np.uint8)
            
            faces = self.detector(np_image)
            if len(faces) == 0:
                self.logger.warning("No face detected in the image")
                return None
            
            landmarks = self.predictor(np_image, faces[0])
            return np.array([[p.x, p.y] for p in landmarks.parts()])
        except Exception as e:
            self.logger.error(f"Error in get_landmarks: {str(e)}")
            return None
    def preprocess_for_face_recognition(self, image):
        try:
            # Resize to 160x160
            image = F.interpolate(image, size=(160, 160), mode='bilinear', align_corners=False)
            
            # Normalize
            mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)
            image = (image - mean) / std
            
            return image
        except Exception as e:
            self.logger.error(f"Error in preprocess_for_face_recognition: {str(e)}")
            return None

    def preprocess_for_emotion(self, image):
        try:
            np_image = self.tensor_to_numpy(image)
            if np_image.ndim == 3:
                np_image = np.transpose(np_image, (1, 2, 0))
            elif np_image.ndim == 2:
                np_image = np.stack((np_image,) * 3, axis=-1)
            else:
                raise ValueError(f"Unexpected number of dimensions: {np_image.ndim}")
            
            np_image = (np_image * 255).astype(np.uint8)
            if np_image.shape[0] == 0 or np_image.shape[1] == 0:
                raise ValueError(f"Invalid image shape: {np_image.shape}")
            np_image = cv2.resize(np_image, (224, 224))
            return np_image
        except Exception as e:
            self.logger.error(f"Error in preprocess_for_emotion: {str(e)}")
            return None

    def preprocess_for_pose_estimation(self, image):
        try:
            # SixDRepNet expects input size of 224x224
            image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
            
            # Normalize using ImageNet mean and std
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            image = (image - mean) / std
            
            return image
        except Exception as e:
            self.logger.error(f"Error in preprocess_for_pose_estimation: {str(e)}")
            return None

    def get_pose(self, image):
        try:
            with torch.no_grad():
                pose_tuple = self.rotation_net.predict(image)
            pose_angles = pose_tuple[0]
            
            # Extract pitch, yaw, roll and ensure they are scalars
            pitch = pose_angles[0, 0].item() if not torch.isnan(pose_angles[0, 0]) else 0.0
            yaw = pose_angles[0, 1].item() if not torch.isnan(pose_angles[0, 1]) else 0.0
            roll = pose_angles[0, 2].item() if not torch.isnan(pose_angles[0, 2]) else 0.0
            
            # Return as tensor
            return torch.tensor([pitch, yaw, roll]).to(self.device)
        except Exception as e:
            self.logger.error(f"Error in get_pose: {str(e)}")
            return torch.tensor([0.0, 0.0, 0.0]).to(self.device)


    def pose_loss(self, pose_real, pose_recon):
        # print("pose_real:", pose_real)
        # print("pose_recon:", pose_recon)
        
        try:
            # Calculate L1 loss for each angle directly on the tensors
            pitch_loss = F.l1_loss(pose_real[0], pose_recon[0])
            yaw_loss = F.l1_loss(pose_real[1], pose_recon[1])
            roll_loss = F.l1_loss(pose_real[2], pose_recon[2])
            
            # Combine losses
            total_pose_loss = pitch_loss + yaw_loss + roll_loss
            
            return total_pose_loss
        except Exception as e:
            self.logger.error(f"Error in pose_loss: {str(e)}")
            return torch.tensor(0.0).to(self.device)

    def landmark_loss(self, x_real, x_recon):
        self.logger.debug(f"x_real shape: {x_real.shape}, x_recon shape: {x_recon.shape}")
        
        batch_size = x_real.size(0)
        self.logger.debug(f"Processing batch of size {batch_size}")
        
        # Pose estimation loss
        pose_real = self.get_pose(x_real)
        print(f"pose_real: {pose_real}")
        pose_recon = self.get_pose(x_recon)
        print(f"pose_recon: {pose_recon}")
        pose_loss = self.pose_loss(pose_real, pose_recon)
        
        total_loss =  self.landmark_weight * pose_loss
        return total_loss

    def emotion_loss(self, fe_s, fe_t, emotion_labels_s, emotion_labels_t):
        # try:
        #     emotion_pred_s = torch.tensor([
        #         self.emotion_model.predict_emotions(self.preprocess_for_emotion(img))[0] 
        #         if self.preprocess_for_emotion(img) is not None else np.zeros(8)
        #         for img in fe_s
        #     ]).to(self.device)
        #     emotion_pred_t = torch.tensor([
        #         self.emotion_model.predict_emotions(self.preprocess_for_emotion(img))[0] 
        #         if self.preprocess_for_emotion(img) is not None else np.zeros(8)
        #         for img in fe_t
        #     ]).to(self.device)
            
        #     emotion_labels_s = emotion_labels_s.float()
        #     emotion_labels_t = emotion_labels_t.float()
            
        #     loss_s = self.ce_loss(emotion_pred_s.float(), emotion_labels_s)
        #     loss_t = self.ce_loss(emotion_pred_t.float(), emotion_labels_t)
        #     total_loss = (loss_s + loss_t) * self.emotion_weight

        #     self.logger.debug(f"Emotion loss - Source: {loss_s:.4f}, Target: {loss_t:.4f}, Total: {total_loss:.4f}")

        #     return total_loss
        # except Exception as e:
        #     self.logger.error(f"Error in emotion_loss: {str(e)}")
        return torch.tensor(0.0).to(self.device)

    def identity_loss(self, fi_s, fi_t):
        try:
            loss = self.l2_loss(fi_s, fi_t)
            self.logger.debug(f"Identity loss: {loss:.4f}")
            return loss
        except Exception as e:
            self.logger.error(f"Error in identity_loss: {str(e)}")
            return torch.tensor(0.0).to(self.device)

    def reconstruction_loss(self, x_s, x_t, x_s_recon, x_t_recon):
        try:
            loss = self.l2_loss(x_s, x_s_recon) + self.l2_loss(x_t, x_t_recon)
            self.logger.debug(f"Reconstruction loss: {loss:.4f}")
            return loss
        except Exception as e:
            self.logger.error(f"Error in reconstruction_loss: {str(e)}")
            return torch.tensor(0.0).to(self.device)

    def forward(self, x_s, x_t, x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, emotion_labels_s, emotion_labels_t):
        try:
            l_landmark = self.landmark_loss(x_s, x_s_recon) + self.landmark_loss(x_t, x_t_recon)
            l_emotion = self.emotion_loss(fe_s, fe_t, emotion_labels_s, emotion_labels_t)
            l_identity = self.identity_loss(fi_s, fi_t)
            l_recon = self.reconstruction_loss(x_s, x_t, x_s_recon, x_t_recon)

            self.logger.debug(f"Loss components - Landmark: {l_landmark:.4f}, Emotion: {l_emotion:.4f}, Identity: {l_identity:.4f}, Recon: {l_recon:.4f}")

            return l_landmark, l_emotion, l_identity, l_recon
        except Exception as e:
            self.logger.error(f"Error in forward pass: {str(e)}")
            raise
