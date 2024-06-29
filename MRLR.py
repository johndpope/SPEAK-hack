import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.nn.functional as F
from model import IRFDGenerator512


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
        self.Ei = self._create_encoder_with_mrlr((2048, 16, 16))
        self.Ee = self._create_encoder_with_mrlr((2048, 16, 16))
        self.Ep = self._create_encoder_with_mrlr((2048, 16, 16))
        

        input_dim = 3 * 1024  # Since you have three encoders (Ei, Ee, Ep)
        self.Gd = IRFDGenerator512(input_dim=input_dim, ngf=64)
        self.Cm = nn.Linear(512, 8)

        self.to(device)

    def _create_encoder_with_mrlr(self, input_shape):
        encoder = resnet50(pretrained=True)
        encoder_layers = list(encoder.children())[:-2]  # Remove avg pool and fc layers
        backbone = nn.Sequential(*encoder_layers)
        # ResNet50 feature output shape: torch.Size([1, 2048, 7, 7])
        # Number of channels: 2048
        # Spatial dimensions: 7x7

        mrlr_extractor = MRLRFeatureExtractor(
        input_shape=(2048, 7, 7),
            partitions=[
                [[0], [1, 2]],  # Separate channels from spatial dimensions
                [[1, 2], [0]],  # Combine spatial dimensions, separate from channels
                [[0, 1], [2]]   # Another view, combining channels with one spatial dimension
            ],
            ranks=[512, 256, 256]  # Adjust these based on your needs
        )
        return nn.Sequential(backbone, mrlr_extractor)


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

