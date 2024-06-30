import torch
import torch.nn as nn
from torchvision.models import resnet50
import colored_traceback.auto
import torch.nn.functional as F
from torchvision import models
import lpips
from torch.utils.checkpoint import checkpoint



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


       

class IRFD(nn.Module):
    def __init__(self, input_dim=2048, ngf=64, max_resolution=256):
        super(IRFD, self).__init__()
        
        self.enable_swapping = True
        # Encoders
        self.Ei = CustomResNet50()  # Identity encoder
        self.Ee = CustomResNet50()  # Emotion encoder
        self.Ep = CustomResNet50()  # Pose encoder
        
        # Generator
        # self.Gd = SimpleGenerator(input_dim=input_dim * 3, ngf=ngf, max_resolution=max_resolution)
        # self.Gd = AnotherGenerator(input_dim=input_dim * 3, ngf=ngf, max_resolution=max_resolution)
        
        self.Gd = BasicGenerator64(input_dim=input_dim * 3)

        self.Cm = nn.Linear(2048, 8) # 8 = num_emotion_classes

    
    def forward(self, x_s, x_t):
        x_s = x_s.requires_grad_(True)
        x_t = x_t.requires_grad_(True)
        # Check input range
        # Clamp input values to [-1, 1] range
        x_s = torch.clamp(x_s, -1, 1)
        x_t = torch.clamp(x_t, -1, 1)

        if x_s.min() < -1 or x_s.max() > 1 or x_t.min() < -1 or x_t.max() > 1:
            print(f"Input range warning: x_s min={x_s.min():.2f}, max={x_s.max():.2f}, x_t min={x_t.min():.2f}, max={x_t.max():.2f}")
    
        # Gradient Checkpointing:
        # Implement gradient checkpointing for the emotion
        fi_s = checkpoint(self.Ei, x_s) 
        fe_s = checkpoint(self.Ee, x_s) 
        fp_s = checkpoint(self.Ep, x_s) 
        fi_t = checkpoint(self.Ei, x_t) 
        fe_t = checkpoint(self.Ee, x_t) 
        fp_t = checkpoint(self.Ep, x_t) 
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"🔥 NaN in {name}")



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
        
        # print("Feature extraction successful!")
        # print(f"Feature shape: {x_s_recon.shape}")
        # print(f"Feature statistics:")
        # print(f"  Min: {x_s_recon.min().item():.4f}")
        # print(f"  Max: {x_s_recon.max().item():.4f}")
        # print(f"  Mean: {x_s_recon.mean().item():.4f}")
        # print(f"  Std: {x_s_recon.std().item():.4f}")

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
    def __init__(self, alpha=0.1, lpips_weight=.3):
        super(IRFDLoss, self).__init__()
        self.alpha = alpha
        self.lpips_weight = lpips_weight
        self.l2_loss = ScaledMSELoss(scale=1)
        self.ce_loss = stable_cross_entropy

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
        
        # 5x (+ lpips)  losses as per paper
        l_identity *= 0.2
        l_cls *= 0.1 # not as important as lpips
        l_pose *= 0.1 # not as important as lpips
        l_emotion *= 0.1 # not as important as lpips
        l_self *= 0.2
        l_lpips *= 0.3 

        
        # Total loss
        total_loss = l_identity + l_cls + l_pose + l_emotion + l_self + l_lpips
        
        # Clip the total loss to prevent extreme values
        total_loss = clip_loss(total_loss)
        
        return total_loss, l_identity, l_cls, l_pose, l_emotion, l_self, l_lpips




'''
Multi-resolution approach: The MRLR method is like looking at a picture at different zoom levels. First, you capture the broad strokes (lower resolution), then you focus on finer details (higher resolution).
PARAFAC decomposition: This is similar to breaking down a complex shape into simpler components. For a 3D tensor, imagine decomposing a complex 3D object into a set of 1D "building blocks" that, when combined, approximate the original object.
Residual updates: This process is like an artist refining a sketch. First, they draw the main outlines, then they focus on the details that are missing from the initial sketch.
Khatri-Rao product: This can be thought of as a way to combine information from different dimensions. It's like creating a "super factor" that represents the interaction between all other factors.
Convergence check: This is similar to a painter stepping back from their canvas to see if the painting looks "close enough" to the subject. If it does, they stop; if not, they continue refining.
'''

# https://github.com/johndpope/MRLR

# https://arxiv.org/pdf/2406.18560
# A MULTI-RESOLUTION LOW-RANK TENSOR DECOMPOSITION


class MRLR:
    def __init__(self, tensor, partitions, ranks):
        self.tensor = tensor
        self.device = tensor.device
        self.dtype = tensor.dtype
        self.partitions = partitions
        self.ranks = ranks
        print(f"Tensor shape: {self.tensor.shape}")
        self._validate_partitions()
        self.factors = self._initialize_factors()

    def _validate_partitions(self):
        tensor_modes = set(range(self.tensor.dim()))
        for i, partition in enumerate(self.partitions):
            partition_modes = set(mode for group in partition for mode in group)
            print(f"Partition {i}: {partition}")
            print(f"Tensor modes: {tensor_modes}")
            print(f"Partition modes: {partition_modes}")
            if partition_modes != tensor_modes:
                print(f"Warning: Partition {i} does not match tensor dimensions. Adjusting partition.")
                self.partitions[i] = self._adjust_partition(partition, tensor_modes)
                print(f"Adjusted Partition {i}: {self.partitions[i]}")

    def _adjust_partition(self, partition, tensor_modes):
        adjusted_partition = []
        remaining_modes = set(tensor_modes)
        for group in partition:
            valid_modes = [mode for mode in group if mode in tensor_modes]
            if valid_modes:
                adjusted_partition.append(valid_modes)
                remaining_modes -= set(valid_modes)
        if remaining_modes:
            adjusted_partition.append(list(remaining_modes))
        return adjusted_partition

    def _initialize_factors(self):
        factors = []
        for partition, rank in zip(self.partitions, self.ranks):
            partition_factors = []
            for mode_group in partition:
                size = 1
                for mode in mode_group:
                    size *= self.tensor.shape[mode]
                partition_factors.append(torch.randn(size, rank, device=self.device, dtype=torch.float32))
            factors.append(partition_factors)
        return factors

    def _unfold(self, tensor, partition):
        modes = [mode for group in partition for mode in group]
        if len(modes) != tensor.dim():
            raise ValueError(f"Number of modes ({len(modes)}) does not match tensor dimensions ({tensor.dim()})")
        permuted = tensor.permute(*modes)
        reshaped = permuted.reshape(permuted.shape[0], -1)
        return reshaped

    def _fold(self, unfolded, partition, original_shape):
        intermediate_shape = []
        for mode_group in partition:
            for mode in mode_group:
                intermediate_shape.append(original_shape[mode])
        return unfolded.reshape(intermediate_shape)

    def _parafac(self, tensor, rank, max_iter=100, tol=1e-4):
        tensor = tensor.to(torch.float32)
        factors = [torch.randn(s, rank, device=self.device, dtype=torch.float32) for s in tensor.shape]
        
        for _ in range(max_iter):
            old_factors = [f.clone() for f in factors]
            for mode in range(len(factors)):
                unfold_mode = self._unfold(tensor, [[mode], list(range(mode)) + list(range(mode+1, tensor.dim()))])
                
                khatri_rao_prod = factors[(mode+1) % len(factors)]
                for i in range(2, len(factors)):
                    current_factor = factors[(mode+i) % len(factors)]
                    khatri_rao_prod = torch.einsum('ir,jr->ijr', khatri_rao_prod, current_factor).reshape(-1, rank)
                
                V = khatri_rao_prod.t() @ khatri_rao_prod
                factor_update = unfold_mode @ khatri_rao_prod
                
                try:
                    factors[mode] = factor_update @ torch.pinverse(V)
                    factors[mode] = torch.nan_to_num(factors[mode], nan=0.0, posinf=1e10, neginf=-1e10)
                except RuntimeError as e:
                    print(f"Error in PARAFAC: {e}")
                    print(f"V shape: {V.shape}")
                    print(f"factor_update shape: {factor_update.shape}")
                    return factors
            
            if all(torch.norm(f - old_f) < tol for f, old_f in zip(factors, old_factors)):
                break
        
        return factors

    def decompose(self, max_iter=100, tol=1e-4):
        residual = self.tensor.to(torch.float32)
        approximations = []

        for partition, rank in zip(self.partitions, self.ranks):
            unfolded = self._unfold(residual, partition)
            
            factors = self._parafac(unfolded, rank, max_iter, tol)
            
            if len(factors) == 2:
                approximation = self._fold(factors[0] @ factors[1].T, partition, residual.shape)
            else:
                approximation = self._fold(self._reconstruct_from_factors(factors), partition, residual.shape)
            
            approximations.append(approximation)
            residual = residual - approximation

        return approximations

    def _reconstruct_from_factors(self, factors):
        reconstructed = factors[0]
        for factor in factors[1:]:
            reconstructed = torch.einsum('...i,ji->...j', reconstructed, factor)
        return reconstructed

    def reconstruct(self):
        result = sum(self.decompose())
        return result.to(self.dtype)

class MRLRFeatureExtractor(nn.Module):
    def __init__(self, input_shape, partitions, ranks):
        super(MRLRFeatureExtractor, self).__init__()
        self.input_shape = input_shape
        self.partitions = partitions
        self.ranks = ranks
        self.mrlr = None  # Will be initialized in forward pass

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Print input shape for debugging
        print(f"MRLRFeatureExtractor input shape: {x.shape}")
        print(f"Expected input shape: {(batch_size, *self.input_shape)}")
        
        # Check if the input shape matches the expected shape
        if x.shape[1:] != self.input_shape:
            print(f"Warning: Input shape {x.shape[1:]} does not match expected shape {self.input_shape}")
            print("Adjusting input_shape to match actual input...")
            self.input_shape = x.shape[1:]
        
        # Reshape the input to match the expected input shape
        x_reshaped = x.view(batch_size, *self.input_shape)
        
        # Initialize MRLR if not already done
        if self.mrlr is None:
            self.mrlr = MRLR(x_reshaped[0], self.partitions, self.ranks)
        
        # Process each item in the batch
        reconstructed = []
        for item in x_reshaped:
            self.mrlr.tensor = item  # Update the tensor in MRLR
            recon = self.mrlr.reconstruct()
            reconstructed.append(recon)
        
        # Stack the reconstructed tensors
        reconstructed = torch.stack(reconstructed)
        
        # Print output shape for debugging
        print(f"MRLRFeatureExtractor output shape: {reconstructed.shape}")
        


class IRFDWithMRLR(nn.Module):
    def __init__(self,device):
        super(IRFDWithMRLR, self).__init__()
        self.Ei = self._create_encoder_with_mrlr((2048, 7, 7))
        self.Ee = self._create_encoder_with_mrlr((2048, 7, 7))
        self.Ep = self._create_encoder_with_mrlr((2048, 7, 7))
        

        input_dim = 3 * 1024  # Since you have three encoders (Ei, Ee, Ep)
        
        self.Gd = BasicGenerator64(input_dim=input_dim * 3)
        self.Cm = nn.Linear(2048, 8) 

        self.to(device)

    def _create_encoder_with_mrlr(self, input_shape):
        model = resnet50(pretrained=True)
        features_extractor = torch.nn.Sequential(*list(model.children())[:-2])
        features_extractor.eval()

        mrlr_extractor = MRLRFeatureExtractor(
        input_shape=(2048, 7, 7),
            partitions=[
                [[0], [1, 2]],  # Separate channels from spatial dimensions
                [[1, 2], [0]],  # Combine spatial dimensions, separate from channels
                [[0, 1], [2]]   # Another view, combining channels with one spatial dimension
            ],
            ranks=[512, 64, 64]  # Adjust these based on your needs
        )
        return nn.Sequential(features_extractor, mrlr_extractor)


    def forward(self, x_s, x_t):
        fi_s = self.Ei(x_s)
        print("fi_s shape:", fi_s.shape())

        fe_s = self.Ee(x_s)
        print("fe_s shape:", fe_s.shape())

        fp_s = self.Ep(x_s)
        print("fp_s shape:", fp_s.shape())

        fi_t = self.Ei(x_t)
        print("fi_t shape:", fi_t.shape())

        fe_t = self.Ee(x_t)
        print("fe_t shape:", fe_t.shape())

        fp_t = self.Ep(x_t)
        print("fp_t shape:", fp_t.shape())
        
        swap_type = torch.randint(0, 3, (1,)).item()
        if swap_type == 0:
            fi_s, fi_t = fi_t, fi_s
        elif swap_type == 1:
            fe_s, fe_t = fe_t, fe_s
        else:
            fp_s, fp_t = fp_t, fp_s
        
        x_s_recon = self.Gd(torch.cat([fi_s, fe_s, fp_s], dim=1))
        x_t_recon = self.Gd(torch.cat([fi_t, fe_t, fp_t], dim=1))
        
        emotion_pred_s = torch.softmax(self.Cm(fe_s), dim=1)
        emotion_pred_t = torch.softmax(self.Cm(fe_t), dim=1)
        
        return x_s_recon, x_t_recon, fi_s, fe_s, fp_s, fi_t, fe_t, fp_t, emotion_pred_s, emotion_pred_t
