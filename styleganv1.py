# coding: UTF-8
"""
    @author: samuel ko
    @date:   2019.04.11
    @notice:
             1) refactor the module of Gsynthesis with
                - LayerEpilogue.
                - Upsample2d.
                - GBlock.
                and etc.
             2) the initialization of every patch we use are all abided by the original NvLabs released code.
             3) Discriminator is a simplicity version of PyTorch.
             4) fix bug: default settings of batchsize.

"""
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import os
from collections import OrderedDict
from torch.nn.init import kaiming_normal_
import logging
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch.nn.utils import spectral_norm


class Blur2d(nn.Module):
    def __init__(self, f=[1,2,1], normalize=True, flip=False, stride=1):
        """
            depthwise_conv2d:
            https://blog.csdn.net/mao_xiao_feng/article/details/78003476
        """
        super(Blur2d, self).__init__()
        assert isinstance(f, list) or f is None, "kernel f must be an instance of python built_in type list!"

        if f is not None:
            f = torch.tensor(f, dtype=torch.float32)
            f = f[:, None] * f[None, :]
            f = f[None, None]
            if normalize:
                f = f / f.sum()
            if flip:
                # f = f[:, :, ::-1, ::-1]
                f = torch.flip(f, [2, 3])
            self.f = f
        else:
            self.f = None
        self.stride = stride

    def forward(self, x):
        if self.f is not None:
            # expand kernel channels
            kernel = self.f.expand(x.size(1), -1, -1, -1).to(x.device)
            x = F.conv2d(
                x,
                kernel,
                stride=self.stride,
                padding=int((self.f.size(2)-1)/2),
                groups=x.size(1)
            )
            return x
        else:
            return x


class Conv2d(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 gain=2 ** (0.5),
                 use_wscale=False,
                 lrmul=1,
                 bias=True):
        super().__init__()
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            return F.conv2d(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul, padding=self.kernel_size // 2)
        else:
            return F.conv2d(x, self.weight * self.w_lrmul, padding=self.kernel_size // 2)


class Upscale2d(nn.Module):
    def __init__(self, factor=2, gain=1):
        """
            the first upsample method in G_synthesis.
        :param factor:
        :param gain:
        """
        super().__init__()
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        if self.gain != 1:
            x = x * self.gain
        if self.factor > 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, self.factor, -1, self.factor)
            x = x.contiguous().view(shape[0], shape[1], self.factor * shape[2], self.factor * shape[3])
        return x


class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp  = torch.mul(x, x) # or x ** 2
        tmp1 = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True) + self.epsilon)

        return x * tmp1


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


class LayerEpilogue(nn.Module):
    def __init__(self,
                 channels,
                 dlatent_size,
                 use_wscale,
                 use_noise,
                 use_pixel_norm,
                 use_instance_norm,
                 use_styles):
        super(LayerEpilogue, self).__init__()

        if use_noise:
            self.noise = ApplyNoise(channels)
        self.act = nn.LeakyReLU(negative_slope=0.2)

        if use_pixel_norm:
            self.pixel_norm = PixelNorm()
        else:
            self.pixel_norm = None

        if use_instance_norm:
            self.instance_norm = InstanceNorm()
        else:
            self.instance_norm = None

        if use_styles:
            self.style_mod = ApplyStyle(dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None

    def forward(self, x, noise, dlatents_in_slice=None):
        x = self.noise(x, noise)
        x = self.act(x)
        if self.pixel_norm is not None:
            x = self.pixel_norm(x)
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)

        return x


class GBlock(nn.Module):
    def __init__(self,
                 res,
                 use_wscale,
                 use_noise,
                 use_pixel_norm,
                 use_instance_norm,
                 noise_input,        # noise
                 dlatent_size=512,   # Disentangled latent (W) dimensionality.
                 use_style=True,     # Enable style inputs?
                 f=None,        # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 factor=2,           # upsample factor.
                 fmap_base=8192,     # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,     # log2 feature map reduction when doubling the resolution.
                 fmap_max=512,       # Maximum number of feature maps in any layer.
                 ):
        super(GBlock, self).__init__()
        self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        # res
        self.res = res

        # blur2d
        self.blur = Blur2d(f)

        # noise
        self.noise_input = noise_input

        if res < 7:
            # upsample method 1
            self.up_sample = Upscale2d(factor)
        else:
            # upsample method 2
            self.up_sample = nn.ConvTranspose2d(self.nf(res-3), self.nf(res-2), 4, stride=2, padding=1)

        # A Composition of LayerEpilogue and Conv2d.
        self.adaIn1 = LayerEpilogue(self.nf(res-2), dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)
        self.conv1  = Conv2d(input_channels=self.nf(res-2), output_channels=self.nf(res-2),
                             kernel_size=3, use_wscale=use_wscale)
        self.adaIn2 = LayerEpilogue(self.nf(res-2), dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)

    def forward(self, x, dlatent):
        x = self.up_sample(x)
        x = self.adaIn1(x, self.noise_input[self.res*2-4], dlatent[:, self.res*2-4])
        x = self.conv1(x)
        x = self.adaIn2(x, self.noise_input[self.res*2-3], dlatent[:, self.res*2-3])
        return x

#model.apply(weights_init)


# =========================================================================
#   Define sub-network
#   2019.3.31
#   FC
# =========================================================================
class G_mapping(nn.Module):
    def __init__(self,
                 mapping_fmaps=512,
                 dlatent_size=512,
                 resolution=1024,
                 normalize_latents=True,  # Normalize latent vectors (Z) before feeding them to the mapping layers?
                 use_wscale=True,         # Enable equalized learning rate?
                 lrmul=0.01,              # Learning rate multiplier for the mapping layers.
                 gain=2**(0.5)            # original gain in tensorflow.
                 ):
        super(G_mapping, self).__init__()
        self.mapping_fmaps = mapping_fmaps
        self.func = nn.Sequential(
            FC(self.mapping_fmaps, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale)
        )

        self.normalize_latents = normalize_latents
        self.resolution_log2 = int(np.log2(resolution))
        self.num_layers = self.resolution_log2 * 2 - 2
        self.pixel_norm = PixelNorm()
        # - 2 means we start from feature map with height and width equals 4.
        # as this example, we get num_layers = 18.

    def forward(self, x):
        if self.normalize_latents:
            x = self.pixel_norm(x)
        out = self.func(x)
        return out, self.num_layers


class G_synthesis(nn.Module):
    def __init__(self,
                 dlatent_size,                       # Disentangled latent (W) dimensionality.
                 resolution=1024,                    # Output resolution (1024 x 1024 by default).
                 fmap_base=8192,                     # Overall multiplier for the number of feature maps.
                 num_channels=3,                     # Number of output color channels.
                 structure='fixed',                  # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
                 fmap_max=512,                       # Maximum number of feature maps in any layer.
                 fmap_decay=1.0,                     # log2 feature map reduction when doubling the resolution.
                 f=None,                        # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 use_pixel_norm      = False,        # Enable pixelwise feature vector normalization?
                 use_instance_norm   = True,        # Enable instance normalization?
                 use_wscale = True,                  # Enable equalized learning rate?
                 use_noise = True,                   # Enable noise inputs?
                 use_style = True                    # Enable style inputs?
                 ):                             # batch size.
        """
            2019.3.31
        :param dlatent_size: 512 Disentangled latent(W) dimensionality.
        :param resolution: 1024 x 1024.
        :param fmap_base:
        :param num_channels:
        :param structure: only support 'fixed' mode.
        :param fmap_max:
        """
        super(G_synthesis, self).__init__()

        self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        self.structure = structure
        self.resolution_log2 = int(np.log2(resolution))
        # - 2 means we start from feature map with height and width equals 4.
        # as this example, we get num_layers = 18.
        num_layers = self.resolution_log2 * 2 - 2
        self.num_layers = num_layers

        # Noise inputs.
        self.noise_inputs = []
        for layer_idx in range(num_layers):
            res = layer_idx // 2 + 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noise_inputs.append(torch.randn(*shape).to("cuda"))

        # Blur2d
        self.blur = Blur2d(f)

        # torgb: fixed mode
        self.channel_shrinkage = Conv2d(input_channels=self.nf(self.resolution_log2-2),
                                        output_channels=self.nf(self.resolution_log2),
                                        kernel_size=3,
                                        use_wscale=use_wscale)
        self.torgb = Conv2d(self.nf(self.resolution_log2), num_channels, kernel_size=1, gain=1, use_wscale=use_wscale)

        # Initial Input Block
        self.const_input = nn.Parameter(torch.ones(1, self.nf(1), 4, 4))
        self.bias = nn.Parameter(torch.ones(self.nf(1)))
        self.adaIn1 = LayerEpilogue(self.nf(1), dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)
        self.conv1  = Conv2d(input_channels=self.nf(1), output_channels=self.nf(1), kernel_size=3, use_wscale=use_wscale)
        self.adaIn2 = LayerEpilogue(self.nf(1), dlatent_size, use_wscale, use_noise, use_pixel_norm,
                                    use_instance_norm, use_style)

        # Common Block
        # 4 x 4 -> 8 x 8
        res = 3
        self.GBlock1 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 8 x 8 -> 16 x 16
        res = 4
        self.GBlock2 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 16 x 16 -> 32 x 32
        res = 5
        self.GBlock3 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 32 x 32 -> 64 x 64
        res = 6
        self.GBlock4 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 64 x 64 -> 128 x 128
        res = 7
        self.GBlock5 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 128 x 128 -> 256 x 256
        res = 8
        self.GBlock6 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 256 x 256 -> 512 x 512
        res = 9
        self.GBlock7 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 512 x 512 -> 1024 x 1024
        res = 10
        self.GBlock8 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

    def forward(self, dlatent):
        """
           dlatent: Disentangled latents (W), shape为[minibatch, num_layers, dlatent_size].
        :param dlatent:
        :return:
        """
        images_out = None
        # Fixed structure: simple and efficient, but does not support progressive growing.
        if self.structure == 'fixed':
            # initial block 0:
            x = self.const_input.expand(dlatent.size(0), -1, -1, -1)
            x = x + self.bias.view(1, -1, 1, 1)
            x = self.adaIn1(x, self.noise_inputs[0], dlatent[:, 0])
            x = self.conv1(x)
            x = self.adaIn2(x, self.noise_inputs[1], dlatent[:, 1])

            # block 1:
            # 4 x 4 -> 8 x 8
            x = self.GBlock1(x, dlatent)

            # block 2:
            # 8 x 8 -> 16 x 16
            x = self.GBlock2(x, dlatent)

            # block 3:
            # 16 x 16 -> 32 x 32
            x = self.GBlock3(x, dlatent)

            # block 4:
            # 32 x 32 -> 64 x 64
            x = self.GBlock4(x, dlatent)

            # block 5:
            # 64 x 64 -> 128 x 128
            x = self.GBlock5(x, dlatent)

            # block 6:
            # 128 x 128 -> 256 x 256
            x = self.GBlock6(x, dlatent)

            # block 7:
            # 256 x 256 -> 512 x 512
            x = self.GBlock7(x, dlatent)

            # block 8:
            # 512 x 512 -> 1024 x 1024
            x = self.GBlock8(x, dlatent)

            x = self.channel_shrinkage(x)
            images_out = self.torgb(x)
            return images_out


class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1) * noise.to(x.device)

class ApplyStyle(nn.Module):
    def __init__(self, latent_size, channels, use_wscale):
        super(ApplyStyle, self).__init__()
        self.linear = FC(latent_size, channels * 2, gain=1.0, use_wscale=use_wscale)

    def forward(self, x, latent):
        style = self.linear(latent)
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x


class FC(nn.Module):
    def __init__(self, in_channels, out_channels, gain=2**(0.5), use_wscale=False, lrmul=1.0, bias=True):
        super(FC, self).__init__()
        he_std = gain * in_channels ** (-0.5)
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
        else:
            out = F.linear(x, self.weight * self.w_lrmul)
        out = F.leaky_relu(out, 0.2, inplace=True)
        return out
    
class StyleGenerator(nn.Module):
    def __init__(self, 
                 input_dim=6144,
                 latent_dim=512,
                 mapping_layers=8, 
                 style_mixing_prob=0.9,       
                 truncation_psi=0.7,         
                 truncation_cutoff=8):        
        super(StyleGenerator, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.style_mixing_prob = style_mixing_prob
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff

        # Mapping network
        layers = []
        for i in range(mapping_layers):
            in_dim = input_dim if i == 0 else latent_dim
            out_dim = latent_dim
            layers.append(FC(in_dim, out_dim, lrmul=0.01, use_wscale=True))
        self.mapping = nn.Sequential(*layers)

        # Synthesis network
        self.synthesis = SynthesisNetwork()

        # We'll create the BatchNorm layer in the forward pass
        self.bn = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def forward(self, features):
        self.logger.debug(f"Input features shape: {features.shape}")
        
        # Map features to w
        w = self.mapping(features)
        self.logger.debug(f"Mapped features (w) shape: {w.shape}")
        
        # Reshape w to match the expected input of the synthesis network
        w = w.unsqueeze(1).repeat(1, self.synthesis.num_layers, 1)
        self.logger.debug(f"Reshaped w shape: {w.shape}")

        # Apply truncation trick
        if self.truncation_psi and self.truncation_cutoff:
            coefs = torch.ones_like(w)
            coefs[:, :self.truncation_cutoff] *= self.truncation_psi
            w = coefs * w
            self.logger.debug("Applied truncation trick")
        
        # Style mixing (during training only)
        if self.training and self.style_mixing_prob > 0:
            if torch.rand(1) < self.style_mixing_prob:
                with torch.no_grad():
                    w2 = self.mapping(torch.randn_like(features))
                    w2 = w2.unsqueeze(1).repeat(1, self.synthesis.num_layers, 1)
                    mix_layer = torch.randint(1, w.size(1), (1,)).item()
                    w[:, mix_layer:] = w2[:, mix_layer:]
                self.logger.debug(f"Applied style mixing at layer {mix_layer}")

        x = self.synthesis(w)

        # Create or update BatchNorm layer if necessary
        if self.bn is None or self.bn.num_features != x.size(1):
            self.bn = nn.BatchNorm2d(x.size(1)).to(x.device)


        x = self.bn(x)

        self.logger.debug(f"Generated image shape: {x.shape}")

        return x

class SynthesisNetwork(nn.Module):
    def __init__(self, resolution=256, fmap_base=8192, fmap_max=512):
        super(SynthesisNetwork, self).__init__()
        self.resolution_log2 = int(np.log2(resolution))
        self.num_layers = self.resolution_log2 * 2 - 2
        
        def nf(stage): return min(int(fmap_base / (2.0 ** stage)), fmap_max)

        self.const_input = nn.Parameter(torch.ones(1, nf(1), 4, 4))
        self.bias = nn.Parameter(torch.zeros(nf(1)))
        self.style_mod = ApplyStyle(512, nf(1), use_wscale=True)
        self.noise_input1 = ApplyNoise(nf(1))

        self.layers = nn.ModuleList()
        for res in range(3, self.resolution_log2 + 1):
            in_channels = nf(res - 2)
            out_channels = nf(res - 1)
            self.layers.append(SynthesisBlock(in_channels, out_channels, res))

        self.to_rgb = nn.Conv2d(nf(self.resolution_log2 - 1), 3, kernel_size=1)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def forward(self, w):
        self.logger.debug(f"Input style tensor shape: {w.shape}")

        x = self.const_input.expand(w.size(0), -1, -1, -1)
        x = x + self.bias.view(1, -1, 1, 1)
        x = self.noise_input1(x, None)
        x = self.style_mod(x, w[:, 0])
        
        self.logger.debug(f"Initial feature map shape: {x.shape}")

        for i, layer in enumerate(self.layers):
            x = layer(x, w[:, i*2+1:i*2+3])
            self.logger.debug(f"After layer {i+1}, shape: {x.shape}")

        x = self.to_rgb(x)
        self.logger.debug(f"Final output shape: {x.shape}")

        return x

class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resolution):
        super(SynthesisBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.noise1 = ApplyNoise(out_channels)
        self.noise2 = ApplyNoise(out_channels)
        self.style_mod1 = ApplyStyle(512, out_channels, use_wscale=True)
        self.style_mod2 = ApplyStyle(512, out_channels, use_wscale=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, w):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.noise1(x, None)
        x = F.leaky_relu(x, 0.2)
        x = self.style_mod1(x, w[:, 0])
        
        x = self.conv2(x)
        x = self.noise2(x, None)
        x = F.leaky_relu(x, 0.2)
        x = self.style_mod2(x, w[:, 1])
        
        return x

class StyleDiscriminator(nn.Module):
    def __init__(self, resolution=256, fmap_base=8192, num_channels=3, fmap_max=512):
        super().__init__()
        self.resolution_log2 = int(np.log2(resolution))
        self.nf = lambda stage: min(int(fmap_base / (2.0 ** stage)), fmap_max)

        self.fromrgb = spectral_norm(nn.Conv2d(num_channels, self.nf(self.resolution_log2-1), kernel_size=1))
        
        self.blocks = nn.ModuleList()
        for res in range(self.resolution_log2, 2, -1):
            in_channels = self.nf(res-1)
            out_channels = self.nf(res-2)
            self.blocks.append(DiscriminatorBlock(in_channels, out_channels))

        self.final_conv = spectral_norm(nn.Conv2d(self.nf(2), self.nf(1), kernel_size=3, padding=1))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense0 = spectral_norm(nn.Linear(self.nf(1), self.nf(0)))
        self.dense1 = spectral_norm(nn.Linear(self.nf(0), 1))
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def nf(self, stage):
        return min(int(self.fmap_base / (2.0 ** stage)), self.fmap_max)
        
    def forward(self, x):
        self.logger.debug(f"Input shape: {x.shape}")
        
        x = F.leaky_relu(self.fromrgb(x), 0.2)
        self.logger.debug(f"After fromrgb: {x.shape}")
        
        for block in self.blocks:
            x = block(x)
            self.logger.debug(f"After block: {x.shape}")
        
        x = F.leaky_relu(self.final_conv(x), 0.2)
        self.logger.debug(f"After final conv: {x.shape}")
        
        x = self.adaptive_pool(x)
        self.logger.debug(f"After adaptive pool: {x.shape}")
        
        x = x.view(x.size(0), -1)
        self.logger.debug(f"After flatten: {x.shape}")
        
        x = F.leaky_relu(self.dense0(x), 0.2)
        x = self.dense1(x)
        
        return x

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2))

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        return x