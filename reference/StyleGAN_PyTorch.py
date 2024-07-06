# coding: UTF-8
"""
    @author: samuel ko
"""
import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms


from networks_stylegan import StyleGenerator, StyleDiscriminator
from networks_gan import Generator, Discriminator
from utils.utils import plotLossCurve
from loss.loss import gradient_penalty, R1Penalty, R2Penalty
from opts.opts import TrainOptions, INFO

from torchvision.utils import save_image
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import random
import torch
import os

# Set random seem for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Hyper-parameters
CRITIC_ITER = 5


def main(opts):
    # Create the data loader
    loader = sunnerData.DataLoader(sunnerData.ImageDataset(
        root=[[opts.path]],
        transform=transforms.Compose([
            sunnertransforms.Resize((1024, 1024)),
            sunnertransforms.ToTensor(),
            sunnertransforms.ToFloat(),
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
            sunnertransforms.Normalize(),
        ])),
        batch_size=opts.batch_size,
        shuffle=True,
    )

    # Create the model
    start_epoch = 0
    G = StyleGenerator()
    D = StyleDiscriminator()

    # Load the pre-trained weight
    if os.path.exists(opts.resume):
        INFO("Load the pre-trained weight!")
        state = torch.load(opts.resume)
        G.load_state_dict(state['G'])
        D.load_state_dict(state['D'])
        start_epoch = state['start_epoch']
    else:
        INFO("Pre-trained weight cannot load successfully, train from scratch!")

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        INFO("Multiple GPU:" + str(torch.cuda.device_count()) + "\t GPUs")
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)
    G.to(opts.device)
    D.to(opts.device)

    # Create the criterion, optimizer and scheduler
    optim_D = optim.Adam(D.parameters(), lr=0.00001, betas=(0.5, 0.999))
    optim_G = optim.Adam(G.parameters(), lr=0.00001, betas=(0.5, 0.999))
    scheduler_D = optim.lr_scheduler.ExponentialLR(optim_D, gamma=0.99)
    scheduler_G = optim.lr_scheduler.ExponentialLR(optim_G, gamma=0.99)

    # Train
    fix_z = torch.randn([opts.batch_size, 512]).to(opts.device)
    softplus = nn.Softplus()
    Loss_D_list = [0.0]
    Loss_G_list = [0.0]
    for ep in range(start_epoch, opts.epoch):
        bar = tqdm(loader)
        loss_D_list = []
        loss_G_list = []
        for i, (real_img,) in enumerate(bar):
            # =======================================================================================================
            #   (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # =======================================================================================================
            # Compute adversarial loss toward discriminator
            D.zero_grad()
            real_img = real_img.to(opts.device)
            real_logit = D(real_img)
            fake_img = G(torch.randn([real_img.size(0), 512]).to(opts.device))
            fake_logit = D(fake_img.detach())
            d_loss = softplus(fake_logit).mean()
            d_loss = d_loss + softplus(-real_logit).mean()

            if opts.r1_gamma != 0.0:
                r1_penalty = R1Penalty(real_img.detach(), D)
                d_loss = d_loss + r1_penalty * (opts.r1_gamma * 0.5)

            if opts.r2_gamma != 0.0:
                r2_penalty = R2Penalty(fake_img.detach(), D)
                d_loss = d_loss + r2_penalty * (opts.r2_gamma * 0.5)

            loss_D_list.append(d_loss.item())

            # Update discriminator
            d_loss.backward()
            optim_D.step()

            # =======================================================================================================
            #   (2) Update G network: maximize log(D(G(z)))
            # =======================================================================================================
            if i % CRITIC_ITER == 0:
                G.zero_grad()
                fake_logit = D(fake_img)
                g_loss = softplus(-fake_logit).mean()
                loss_G_list.append(g_loss.item())

                # Update generator
                g_loss.backward()
                optim_G.step()

            # Output training stats
            bar.set_description("Epoch {} [{}, {}] [G]: {} [D]: {}".format(ep, i+1, len(loader), loss_G_list[-1], loss_D_list[-1]))

        # Save the result
        Loss_G_list.append(np.mean(loss_G_list))
        Loss_D_list.append(np.mean(loss_D_list))

        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            fake_img = G(fix_z).detach().cpu()
            save_image(fake_img, os.path.join(opts.det, 'images', str(ep) + '.png'), nrow=4, normalize=True)

        # Save model
        state = {
            'G': G.state_dict(),
            'D': D.state_dict(),
            'Loss_G': Loss_G_list,
            'Loss_D': Loss_D_list,
            'start_epoch': ep,
        }
        torch.save(state, os.path.join(opts.det, 'models', 'latest.pth'))

        scheduler_D.step()
        scheduler_G.step()

    # Plot the total loss curve
    Loss_D_list = Loss_D_list[1:]
    Loss_G_list = Loss_G_list[1:]
    plotLossCurve(opts, Loss_D_list, Loss_G_list)


if __name__ == '__main__':
    opts = TrainOptions().parse()
    main(opts)

# coding: UTF-8
"""
    @author: samuel ko
"""
import argparse
import torch
import os


def INFO(inputs):
    print("[ Style GAN ] %s" % (inputs))


def presentParameters(args_dict):
    """
        Print the parameters setting line by line

        Arg:    args_dict   - The dict object which is transferred from argparse Namespace object
    """
    INFO("========== Parameters ==========")
    for key in sorted(args_dict.keys()):
        INFO("{:>15} : {}".format(key, args_dict[key]))
    INFO("===============================")


class TrainOptions():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--path', type=str, default='./star/')
        parser.add_argument('--epoch', type=int, default=500)
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--type', type=str, default='style')
        parser.add_argument('--resume', type=str, default='train_result/models/latest.pth')
        parser.add_argument('--det', type=str, default='train_result')
        parser.add_argument('--r1_gamma', type=float, default=10.0)
        parser.add_argument('--r2_gamma', type=float, default=0.0)
        self.opts = parser.parse_args()

    def parse(self):
        self.opts.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Check if the parameter is valid
        if self.opts.type not in ['style', 'origin']:
            raise Exception(
                "Unknown type: {}  You should assign one of them ['style', 'origin']...".format(self.opts.type))

        # Create the destination folder
        if not os.path.exists(self.opts.det):
            os.mkdir(self.opts.det)
        if not os.path.exists(os.path.join(self.opts.det, 'images')):
            os.mkdir(os.path.join(self.opts.det, 'images'))
        if not os.path.exists(os.path.join(self.opts.det, 'models')):
            os.mkdir(os.path.join(self.opts.det, 'models'))

        # Print the options
        presentParameters(vars(self.opts))
        return self.opts


class InferenceOptions():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--resume', type=str, default='train_result/model/latest.pth')
        parser.add_argument('--type', type=str, default='style')
        parser.add_argument('--num_face', type=int, default=32)
        parser.add_argument('--det', type=str, default='result.png')
        self.opts = parser.parse_args()

    def parse(self):
        self.opts.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Print the options
        presentParameters(vars(self.opts))
        return self.opts

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


class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1) * noise.to(x.device)


class ApplyStyle(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """
    def __init__(self, latent_size, channels, use_wscale):
        super(ApplyStyle, self).__init__()
        self.linear = FC(latent_size,
                      channels * 2,
                      gain=1.0,
                      use_wscale=use_wscale)

    def forward(self, x, latent):
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)    # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x


class FC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 gain=2**(0.5),
                 use_wscale=False,
                 lrmul=1.0,
                 bias=True):
        """
            The complete conversion of Dense/FC/Linear Layer of original Tensorflow version.
        """
        super(FC, self).__init__()
        he_std = gain * in_channels ** (-0.5)  # He init
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


class StyleGenerator(nn.Module):
    def __init__(self,
                 mapping_fmaps=512,
                 style_mixing_prob=0.9,       # Probability of mixing styles during training. None = disable.
                 truncation_psi=0.7,          # Style strength multiplier for the truncation trick. None = disable.
                 truncation_cutoff=8,          # Number of layers for which to apply the truncation trick. None = disable.
                 **kwargs
                 ):
        super(StyleGenerator, self).__init__()
        self.mapping_fmaps = mapping_fmaps
        self.style_mixing_prob = style_mixing_prob
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff

        self.mapping = G_mapping(self.mapping_fmaps, **kwargs)
        self.synthesis = G_synthesis(self.mapping_fmaps, **kwargs)

    def forward(self, latents1):
        dlatents1, num_layers = self.mapping(latents1)
        # let [N, O] -> [N, num_layers, O]
        # 这里的unsqueeze不能使用inplace操作, 如果这样的话, 反向传播的链条会断掉的.
        dlatents1 = dlatents1.unsqueeze(1)
        dlatents1 = dlatents1.expand(-1, int(num_layers), -1)

        # Add mixing style mechanism.
        # with torch.no_grad():
        #     latents2 = torch.randn(latents1.shape).to(latents1.device)
        #     dlatents2, num_layers = self.mapping(latents2)
        #     dlatents2 = dlatents2.unsqueeze(1)
        #     dlatents2 = dlatents2.expand(-1, int(num_layers), -1)
        #
        #     # TODO: original NvLABs produce a placeholder "lod", this mechanism was not added here.
        #     cur_layers = num_layers
        #     mix_layers = num_layers
        #     if np.random.random() < self.style_mixing_prob:
        #         mix_layers = np.random.randint(1, cur_layers)
        #
        #     # NvLABs: dlatents = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, tf.shape(dlatents)), dlatents, dlatents2)
        #     for i in range(num_layers):
        #         if i >= mix_layers:
        #             dlatents1[:, i, :] = dlatents2[:, i, :]

        # Apply truncation trick.
        if self.truncation_psi and self.truncation_cutoff:
            coefs = np.ones([1, num_layers, 1], dtype=np.float32)
            for i in range(num_layers):
                if i < self.truncation_cutoff:
                    coefs[:, i, :] *= self.truncation_psi
            """Linear interpolation.
               a + (b - a) * t (a = 0)
               reduce to
               b * t
            """

            dlatents1 = dlatents1 * torch.Tensor(coefs).to(dlatents1.device)

        img = self.synthesis(dlatents1)
        return img


class StyleDiscriminator(nn.Module):
    def __init__(self,
                 resolution=1024,
                 fmap_base=8192,
                 num_channels=3,
                 structure='fixed',  # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, only support 'fixed' mode now.
                 fmap_max=512,
                 fmap_decay=1.0,
                 # f=[1, 2, 1]         # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 f=None         # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 ):
        """
            Noitce: we only support input pic with height == width.

            if H or W >= 128, we use avgpooling2d to do feature map shrinkage.
            else: we use ordinary conv2d.
        """
        super().__init__()
        self.resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** self.resolution_log2 and resolution >= 4
        self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        # fromrgb: fixed mode
        self.fromrgb = nn.Conv2d(num_channels, self.nf(self.resolution_log2-1), kernel_size=1)
        self.structure = structure

        # blur2d
        self.blur2d = Blur2d(f)

        # down_sample
        self.down1 = nn.AvgPool2d(2)
        self.down21 = nn.Conv2d(self.nf(self.resolution_log2-5), self.nf(self.resolution_log2-5), kernel_size=2, stride=2)
        self.down22 = nn.Conv2d(self.nf(self.resolution_log2-6), self.nf(self.resolution_log2-6), kernel_size=2, stride=2)
        self.down23 = nn.Conv2d(self.nf(self.resolution_log2-7), self.nf(self.resolution_log2-7), kernel_size=2, stride=2)
        self.down24 = nn.Conv2d(self.nf(self.resolution_log2-8), self.nf(self.resolution_log2-8), kernel_size=2, stride=2)

        # conv1: padding=same
        self.conv1 = nn.Conv2d(self.nf(self.resolution_log2-1), self.nf(self.resolution_log2-1), kernel_size=3, padding=(1, 1))
        self.conv2 = nn.Conv2d(self.nf(self.resolution_log2-1), self.nf(self.resolution_log2-2), kernel_size=3, padding=(1, 1))
        self.conv3 = nn.Conv2d(self.nf(self.resolution_log2-2), self.nf(self.resolution_log2-3), kernel_size=3, padding=(1, 1))
        self.conv4 = nn.Conv2d(self.nf(self.resolution_log2-3), self.nf(self.resolution_log2-4), kernel_size=3, padding=(1, 1))
        self.conv5 = nn.Conv2d(self.nf(self.resolution_log2-4), self.nf(self.resolution_log2-5), kernel_size=3, padding=(1, 1))
        self.conv6 = nn.Conv2d(self.nf(self.resolution_log2-5), self.nf(self.resolution_log2-6), kernel_size=3, padding=(1, 1))
        self.conv7 = nn.Conv2d(self.nf(self.resolution_log2-6), self.nf(self.resolution_log2-7), kernel_size=3, padding=(1, 1))
        self.conv8 = nn.Conv2d(self.nf(self.resolution_log2-7), self.nf(self.resolution_log2-8), kernel_size=3, padding=(1, 1))

        # calculate point:
        self.conv_last = nn.Conv2d(self.nf(self.resolution_log2-8), self.nf(1), kernel_size=3, padding=(1, 1))
        self.dense0 = nn.Linear(fmap_base, self.nf(0))
        self.dense1 = nn.Linear(self.nf(0), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        if self.structure == 'fixed':
            x = F.leaky_relu(self.fromrgb(input), 0.2, inplace=True)
            # 1. 1024 x 1024 x nf(9)(16) -> 512 x 512
            res = self.resolution_log2
            x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)

            # 2. 512 x 512 -> 256 x 256
            res -= 1
            x = F.leaky_relu(self.conv2(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)

            # 3. 256 x 256 -> 128 x 128
            res -= 1
            x = F.leaky_relu(self.conv3(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)

            # 4. 128 x 128 -> 64 x 64
            res -= 1
            x = F.leaky_relu(self.conv4(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)

            # 5. 64 x 64 -> 32 x 32
            res -= 1
            x = F.leaky_relu(self.conv5(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down21(self.blur2d(x)), 0.2, inplace=True)

            # 6. 32 x 32 -> 16 x 16
            res -= 1
            x = F.leaky_relu(self.conv6(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down22(self.blur2d(x)), 0.2, inplace=True)

            # 7. 16 x 16 -> 8 x 8
            res -= 1
            x = F.leaky_relu(self.conv7(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down23(self.blur2d(x)), 0.2, inplace=True)

            # 8. 8 x 8 -> 4 x 4
            res -= 1
            x = F.leaky_relu(self.conv8(x), 0.2, inplace=True)
            x = F.leaky_relu(self.down24(self.blur2d(x)), 0.2, inplace=True)

            # 9. 4 x 4 -> point
            x = F.leaky_relu(self.conv_last(x), 0.2, inplace=True)
            # N x 8192(4 x 4 x nf(1)).
            x = x.view(x.size(0), -1)
            x = F.leaky_relu(self.dense0(x), 0.2, inplace=True)
            # N x 1
            x = F.leaky_relu(self.dense1(x), 0.2, inplace=True)
            return x


from torchvision_sunner.constant import *

"""
    This script defines the function which are widely used in the whole package

    Author: SunnerLi
"""

def quiet():
    """
        Mute the information toward the whole log in the toolkit
    """
    global verbose
    verbose = False

def INFO(string = None):
    """
        Print the information with prefix

        Arg:    string  - The string you want to print
    """
    if verbose:
        if string:
            print("[ Torchvision_sunner ] %s" % (string))
        else:
            print("[ Torchvision_sunner ] " + '=' * 50)
from collections import Counter
from PIL import Image
from glob import glob
import numpy as np
import os

"""
    This script defines the function to read the containing of folder and read the file
    You should customize if your data is not considered in the torchvision_sunner previously

    Author: SunnerLi
"""

def readContain(folder_name):
    """
        Read the containing in the particular folder

        ==================================================================
        You should customize this function if your data is not considered
        ==================================================================

        Arg:    folder_name - The path of folder
        Ret:    The list of containing
    """
    # Check the common type in the folder
    common_type = Counter()
    for name in os.listdir(folder_name):
        common_type[name.split('.')[-1]] += 1
    common_type = common_type.most_common()[0][0]

    # Deal with the type
    if common_type == 'jpg':
        name_list = glob(os.path.join(folder_name, '*.jpg'))
    elif common_type == 'png':
        name_list = glob(os.path.join(folder_name, '*.png'))
    elif common_type == 'mp4':
        name_list = glob(os.path.join(folder_name, '*.mp4'))
    else:
        raise Exception("Unknown type {}, You should customize in read.py".format(common_type))
    return name_list

def readItem(item_name):
    """
        Read the file for the given item name

        ==================================================================
        You should customize this function if your data is not considered
        ==================================================================

        Arg:    item_name   - The path of the file
        Ret:    The item you read
    """
    file_type = item_name.split('.')[-1]
    if file_type == "png" or file_type == 'jpg':
        file_obj = np.asarray(Image.open(item_name))
        
        if len(file_obj.shape) == 3:
            # Ignore the 4th dim (RGB only)
            file_obj = file_obj[:, :, :3]
        elif len(file_obj.shape) == 2:
            # Make the rank of gray-scale image as 3
            file_obj = np.expand_dims(file_obj, axis = -1)
    return file_obj

from torchvision_sunner.data.base_dataset import BaseDataset
from torchvision_sunner.read import readContain, readItem
from torchvision_sunner.constant import *
from torchvision_sunner.utils import INFO

from skimage import io as io
# from PIL import Image
from glob import glob

import torch.utils.data as Data

import pickle
import math
import os

"""
    This script define the structure of image dataset

    =======================================================================================
    In the new version, we accept the form that the combination of image and folder:
    e.g. [[image1.jpg, image_folder]]
    On the other hand, the root can only be 'the list of list'
    You should use double list to represent different image domain.
    For example:
        [[image1.jpg], [image2.jpg]]                                => valid
        [[image1.jpg], [image_folder]]                              => valid
        [[image1.jpg, image2.jpg], [image_folder1, image_folder2]]  => valid
        [image1.jpg, image2.jpg]                                    => invalid!
    Also, the triple of nested list is not allow
    =======================================================================================

    Author: SunnerLi
"""

class ImageDataset(BaseDataset):
    def __init__(self, root = None, file_name = '.remain.pkl', sample_method = UNDER_SAMPLING, transform = None, 
                    split_ratio = 0.0, save_file = False):
        """
            The constructor of ImageDataset

            Arg:    root            - The list object. The image set
                    file_name       - The str. The name of record file. 
                    sample_method   - sunnerData.UNDER_SAMPLING or sunnerData.OVER_SAMPLING. Use down sampling or over sampling to deal with data unbalance problem.
                                      (default is sunnerData.OVER_SAMPLING)
                    transform       - transform.Compose object. You can declare some pre-process toward the image
                    split_ratio     - Float. The proportion to split the data. Usually used to split the testing data
                    save_file       - Bool. If storing the record file or not. Default is False
        """
        super().__init__()
        # Record the parameter
        self.root = root
        self.file_name = file_name
        self.sample_method = sample_method
        self.transform = transform
        self.split_ratio = split_ratio
        self.save_file = save_file
        self.img_num = -1
        INFO()

        # Substitude the contain of record file if the record file is exist
        if os.path.exists(file_name) and self.loadFromFile(file_name):
            self.getImgNum()
        elif not os.path.exists(file_name) and root is None:
            raise Exception("Record file {} not found. You should assign 'root' parameter!".format(file_name))
        else:   
            # Extend the images of folder into domain list
            self.getFiles()

            # Change root obj as the index format
            self.root = range(len(self.root))

            # Adjust the image number
            self.getImgNum()

            # Split the files if split_ratio is more than 0.0
            self.split()       

            # Save the split information
            self.save() 

        # Print the domain information
        self.print()

    # ===========================================================================================
    #       Define IO function
    # ===========================================================================================
    def loadFromFile(self, file_name):
        """
            Load the root and files information from .pkl record file
            This function will return False if the record file format is invalid

            Arg:    file_name   - The name of record file
            Ret:    If the loading procedure are successful or not
        """
        return super().loadFromFile(file_name, 'image')

    def save(self, split_file_name = ".split.pkl"):
        """
            Save the information into record file

            Arg:    split_file_name - The path of record file which store the information of split data
        """
        super().save(self.file_name, self.split_ratio, split_file_name, 'image')

    # ===========================================================================================
    #       Define main function
    # ===========================================================================================
    def getFiles(self):
        """
            Construct the files object for the assigned root
            We accept the user to mix folder with image
            This function can extract whole image in the folder
            The element in the files will all become image 

            *******************************************************
            * This function only work if the files object is None *
            *******************************************************
        """
        if not self.files:
            self.files = {}
            for domain_idx, domain in enumerate(self.root):
                images = []
                for img in domain:
                    if os.path.exists(img):
                        if os.path.isdir(img):
                            images += readContain(img)
                        else:
                            images.append(img)
                    else:
                        raise Exception("The path {} is not exist".format(img))
                self.files[domain_idx] = sorted(images)

    def getImgNum(self):
        """
            Obtain the image number in the loader for the specific sample method
            The function will check if the folder has been extracted
        """
        if self.img_num == -1:
            # Check if the folder has been extracted
            for domain in self.root:
                for img in self.files[domain]:
                    if os.path.isdir(img):
                        raise Exception("You should extend the image in the folder {} first!" % img)

            # Statistic the image number
            for domain in self.root:
                if domain == 0:
                    self.img_num = len(self.files[domain])
                else:
                    if self.sample_method == OVER_SAMPLING:
                        self.img_num = max(self.img_num, len(self.files[domain]))
                    elif self.sample_method == UNDER_SAMPLING:
                        self.img_num = min(self.img_num, len(self.files[domain]))
        return self.img_num

    def split(self):
        """
            Split the files object into split_files object
            The original files object will shrink

            We also consider the case of pair image
            Thus we will check if the number of image in each domain is the same
            If it does, then we only generate the list once
        """
        # Check if the number of image in different domain is the same
        if not self.files:
            self.getFiles()
        pairImage = True
        for domain in range(len(self.root) - 1):
            if len(self.files[domain]) != len(self.files[domain + 1]):
                pairImage = False

        # Split the files
        self.split_files = {}
        if pairImage:
            split_img_num = math.floor(len(self.files[0]) * self.split_ratio)
            choice_index_list = self.generateIndexList(range(len(self.files[0])), size = split_img_num)
        for domain in range(len(self.root)):
            # determine the index list
            if not pairImage:
                split_img_num = math.floor(len(self.files[domain]) * self.split_ratio)
                choice_index_list = self.generateIndexList(range(len(self.files[domain])), size = split_img_num)
            # remove the corresponding term and add into new list
            split_img_list = []
            remain_img_list = self.files[domain].copy()
            for j in choice_index_list:
                split_img_list.append(self.files[domain][j])
            for j in choice_index_list:
                self.files[domain].remove(remain_img_list[j])
            self.split_files[domain] = sorted(split_img_list)

    def print(self):
        """
            Print the information for each image domain
        """
        INFO()
        for domain in range(len(self.root)):
            INFO("domain index: %d \timage number: %d" % (domain, len(self.files[domain])))
        INFO()

    def __len__(self):
        return self.img_num

    def __getitem__(self, index):
        return_list = []
        for domain in self.root:
            img_path = self.files[domain][index]
            img = readItem(img_path)
            if self.transform:
                img = self.transform(img)
            return_list.append(img)
        return return_list
from torchvision_sunner.data.base_dataset import BaseDataset
from torchvision_sunner.read import readContain, readItem
from torchvision_sunner.constant import *
from torchvision_sunner.utils import INFO

import torch.utils.data as Data

from PIL import Image
from glob import glob
import numpy as np
import subprocess
import random
import pickle
import torch
import math
import os

"""
    This script define the structure of video dataset

    =======================================================================================
    In the new version, we accept the form that the combination of video and folder:
    e.g. [[video1.mp4, image_folder]]
    On the other hand, the root can only be 'the list of list'
    You should use double list to represent different image domain.
    For example:
        [[video1.mp4], [video2.mp4]]                                => valid
        [[video1.mp4], [video_folder]]                              => valid
        [[video1.mp4, video2.mp4], [video_folder1, video_folder2]]  => valid
        [video1.mp4, video2.mp4]                                    => invalid!
    Also, the triple of nested list is not allow
    =======================================================================================

    Author: SunnerLi
"""

class VideoDataset(BaseDataset):
    def __init__(self, root = None, file_name = '.remain.pkl', T = 10, sample_method = UNDER_SAMPLING, transform = None, 
                    split_ratio = 0.0, decode_root = './.decode', save_file = False):
        """
            The constructor of VideoDataset

            Arg:    root            - The list object. The image set
                    file_name       - Str. The name of record file.
                    T               - Int. The maximun length of small video sequence
                    sample_method   - sunnerData.UNDER_SAMPLING or sunnerData.OVER_SAMPLING. Use down sampling or over sampling to deal with data unbalance problem.
                                      (default is sunnerData.OVER_SAMPLING)
                    transform       - transform.Compose object. You can declare some pre-process toward the image
                    split_ratio     - Float. The proportion to split the data. Usually used to split the testing data
                    decode_root     - Str. The path to store the ffmpeg decode result. 
                    save_file       - Bool. If storing the record file or not. Default is False
        """
        super().__init__()

        # Record the parameter
        self.root = root
        self.file_name = file_name
        self.T = T
        self.sample_method = sample_method
        self.transform = transform
        self.split_ratio = split_ratio
        self.decode_root = decode_root
        self.video_num = -1
        self.split_root  = None
        INFO()

        # Substitude the contain of record file if the record file is exist
        if not os.path.exists(file_name) and root is None:
            raise Exception("Record file {} not found. You should assign 'root' parameter!".format(file_name))
        elif os.path.exists(file_name):
            INFO("Load from file: {}".format(file_name))
            self.loadFromFile(file_name)      

        # Extend the images of folder into domain list
        self.extendFolder()

        # Split the image
        self.split()

        # Form the files obj
        self.getFiles()

        # Adjust the image number
        self.getVideoNum()

        # Save the split information
        self.save() 

        # Print the domain information
        self.print()

    # ===========================================================================================
    #       Define IO function
    # ===========================================================================================
    def loadFromFile(self, file_name):
        """
            Load the root and files information from .pkl record file
            This function will return False if the record file format is invalid

            Arg:    file_name   - The name of record file
            Ret:    If the loading procedure are successful or not
        """        
        return super().loadFromFile(file_name, 'video')

    def save(self, split_file_name = ".split.pkl"):
        """
            Save the information into record file

            Arg:    split_file_name - The path of record file which store the information of split data
        """
        super().save(self.file_name, self.split_ratio, split_file_name, 'video')

    # ===========================================================================================
    #       Define main function
    # ===========================================================================================
    def to_folder(self, name):
        """
            Transfer the name into the folder format
            e.g. 
                '/home/Dataset/video1_folder' => 'home_Dataset_video1_folder'
                '/home/Dataset/video1.mp4'    => 'home_Dataset_video1'

            Arg:    name    - Str. The path of file or original folder
            Ret:    The new (encoded) folder name
        """
        if not os.path.isdir(name):
            name = '_'.join(name.split('.')[:-1]) 
        domain_list = name.split('/')
        while True:
            if '.' in domain_list:
                domain_list.remove('.')
            elif '..' in domain_list:
                domain_list.remove('..')
            else:
                break
        return '_'.join(domain_list)

    def extendFolder(self):
        """
            Extend the video folder in root obj
        """
        if not self.files:
            # Extend the folder of video and replace as new root obj
            extend_root = []
            for domain in self.root:
                videos = []
                for video in domain:
                    if os.path.exists(video):
                        if os.path.isdir(video):
                            videos += readContain(video)
                        else:
                            videos.append(video)
                    else:
                        raise Exception("The path {} is not exist".format(videos))
                extend_root.append(videos)
            self.root = extend_root

    def split(self):
        """
            Split the root object into split_root object
            The original root object will shrink

            We also consider the case of pair image
            Thus we will check if the number of image in each domain is the same
            If it does, then we only generate the list once
        """
        # Check if the number of video in different domain is the same
        pairImage = True
        for domain_idx in range(len(self.root) - 1):
            if len(self.root[domain_idx]) != len(self.root[domain_idx + 1]):
                pairImage = False

        # Split the files
        self.split_root = []
        if pairImage:
            split_img_num = math.floor(len(self.root[0]) * self.split_ratio)
            choice_index_list = self.generateIndexList(range(len(self.root[0])), size = split_img_num)
        for domain_idx in range(len(self.root)):
            # determine the index list
            if not pairImage:
                split_img_num = math.floor(len(self.root[domain_idx]) * self.split_ratio)
                choice_index_list = self.generateIndexList(range(len(self.root[domain_idx])), size = split_img_num)
            # remove the corresponding term and add into new list
            split_img_list = []
            remain_img_list = self.root[domain_idx].copy()
            for j in choice_index_list:
                split_img_list.append(self.root[domain_idx][j])
            for j in choice_index_list:
                self.root[domain_idx].remove(remain_img_list[j])
            self.split_root.append(sorted(split_img_list))

    def getFiles(self):
        """
            Construct the files object for the assigned root
            We accept the user to mix folder with image
            This function can extract whole image in the folder
            
            However, unlike the setting in ImageDataset, we store the video result in root obj.
            Also, the 'images' name will be store in files obj

            The following list the progress of this function:
                1. check if we need to decode again
                2. decode if needed
                3. form the files obj
        """
        if not self.files:
            # Check if the decode process should be conducted again
            should_decode = not os.path.exists(self.decode_root)
            if not should_decode:
                for domain_idx, domain in enumerate(self.root):
                    for video in domain:
                        if not os.path.exists(os.path.join(self.decode_root, str(domain_idx), self.to_folder(video))):
                            should_decode = True
                            break

            # Decode the video if needed
            if should_decode:
                INFO("Decode from scratch...")
                if os.path.exists(self.decode_root):
                    subprocess.call(['rm', '-rf', self.decode_root])
                os.mkdir(self.decode_root)
                self.decodeVideo()
            else:
                INFO("Skip the decode process!")                

            # Form the files object
            self.files = {}
            for domain_idx, domain in enumerate(os.listdir(self.decode_root)):
                self.files[domain_idx] = []
                for video in os.listdir(os.path.join(self.decode_root, domain)):
                    self.files[domain_idx] += [
                        sorted(glob(os.path.join(self.decode_root, domain, video, "*")))
                    ]

    def decodeVideo(self):
        """
            Decode the single video into a series of images, and store into particular folder
        """
        for domain_idx, domain in enumerate(self.root):
            decode_domain_folder = os.path.join(self.decode_root, str(domain_idx))
            os.mkdir(decode_domain_folder)
            for video in domain:
                os.mkdir(os.path.join(self.decode_root, str(domain_idx), self.to_folder(video)))
                source = os.path.join(domain, video)
                target = os.path.join(decode_domain_folder, self.to_folder(video), "%5d.png")
                subprocess.call(['ffmpeg', '-i', source, target])

    def getVideoNum(self):
        """
            Obtain the video number in the loader for the specific sample method
            The function will check if the folder has been extracted
        """
        if self.video_num == -1:
            # Check if the folder has been extracted
            for domain in self.root:
                for video in domain:
                    if os.path.isdir(video):
                        raise Exception("You should extend the image in the folder {} first!" % video)

            # Statistic the image number
            for i, domain in enumerate(self.root):
                if i == 0:
                    self.video_num = len(domain)
                else:
                    if self.sample_method == OVER_SAMPLING:
                        self.video_num = max(self.video_num, len(domain))
                    elif self.sample_method == UNDER_SAMPLING:
                        self.video_num = min(self.video_num, len(domain))
        return self.video_num

    def print(self):
        """
            Print the information for each image domain
        """
        INFO()
        for domain in range(len(self.root)):
            total_frame = 0
            for video in self.files[domain]:
                total_frame += len(video)
            INFO("domain index: %d \tvideo number: %d\tframe total: %d" % (domain, len(self.root[domain]), total_frame))
        INFO()

    def __len__(self):
        return self.video_num

    def __getitem__(self, index):
        """
            Return single batch of data, and the rank is BTCHW
        """
        result = []
        for domain_idx in range(len(self.root)):

            # Form the sequence in single domain
            film_sequence = []
            max_init_frame_idx = len(self.files[domain_idx][index]) - self.T
            start_pos = random.randint(0, max_init_frame_idx)
            for i in range(self.T):
                img_path = self.files[domain_idx][index][start_pos + i]
                img = readItem(img_path)
                film_sequence.append(img)

            # Transform the film sequence
            film_sequence = np.asarray(film_sequence)
            if self.transform:
                film_sequence = self.transform(film_sequence)
            result.append(film_sequence)
        return result
"""
    This script define the wrapper of the Torchvision_sunner.data

    Author: SunnerLi
"""
from torch.utils.data import DataLoader

from torchvision_sunner.data.image_dataset import ImageDataset
from torchvision_sunner.data.video_dataset import VideoDataset
from torchvision_sunner.data.loader import *
from torchvision_sunner.constant import *
from torchvision_sunner.utils import *
from torchvision_sunner.constant import *
from collections import Iterator
import torch.utils.data as data

"""
    This script define the extra loader, and it can be used in flexibility. The loaders include:
        1. ImageLoader (The old version exist)
        2. MultiLoader
        3. IterationLoader

    Author: SunnerLi
"""

class ImageLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers = 1):
        """
            The DataLoader object which can deal with ImageDataset object.

            Arg:    dataset     - ImageDataset. You should use sunnerData.ImageDataset to generate the instance first
                    batch_size  - Int.
                    shuffle     - Bool. Shuffle the data or not
                    num_workers - Int. How many thread you want to use to read the batch data
        """
        super(ImageLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers = num_workers)
        self.dataset = dataset
        self.iter_num = self.__len__()

    def __len__(self):       
        return round(self.dataset.img_num / self.batch_size)

    def getImageNumber(self):
        return self.dataset.img_num

class MultiLoader(Iterator):
    def __init__(self, datasets, batch_size=1, shuffle=False, num_workers = 1):
        """
            This class can deal with multiple dataset object

            Arg:    datasets    - The list of ImageDataset.
                    batch_size  - Int.
                    shuffle     - Bool. Shuffle the data or not
                    num_workers - Int. How many thread you want to use to read the batch data
        """
        # Create loaders
        self.loaders = []
        for dataset in datasets:
            self.loaders.append(
                data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
            )
        
        # Check the sample method
        self.sample_method = None
        for dataset in datasets:
            if self.sample_method is None:
                self.sample_method = dataset.sample_method
            else:
                if self.sample_method != dataset.sample_method:
                    raise Exception("Sample methods are not consistant, {} <=> {}".format(
                        self.sample_method, dataset.sample_method
                    ))

        # Check the iteration number 
        self.iter_num = 0
        for i, dataset in enumerate(datasets):
            if i == 0:
                self.iter_num = len(dataset)
            else:
                if self.sample_method == UNDER_SAMPLING:
                    self.iter_num = min(self.iter_num, len(dataset))
                else:
                    self.iter_num = max(self.iter_num, len(dataset))
        self.iter_num = round(self.iter_num / batch_size)

    def __len__(self):
        return self.iter_num

    def __iter__(self):
        self.iter_loaders = []
        for loader in self.loaders:
            self.iter_loaders.append(iter(loader))
        return self

    def __next__(self):
        result = []
        for loader in self.iter_loaders:
            for _ in loader.__next__():
                result.append(_)
        return tuple(result)

class IterationLoader(Iterator):
    def __init__(self, loader, max_iter = 1):
        """
            Constructor of the loader with specific iteration (not epoch)
            The iteration object will create again while getting end
            
            Arg:    loader      - The torch.data.DataLoader object
                    max_iter    - The maximun iteration
        """
        super().__init__()
        self.loader = loader
        self.loader_iter = iter(self.loader)
        self.iter = 0
        self.max_iter = max_iter

    def __next__(self):
        try:
            result_tuple = next(self.loader_iter)
        except:
            self.loader_iter = iter(self.loader)
            result_tuple = next(self.loader_iter)
        self.iter += 1
        if self.iter <= self.max_iter:
            return result_tuple
        else:
            print("", end='')
            raise StopIteration()

    def __len__(self):
        return self.max_iter
from torchvision_sunner.constant import *
from torchvision_sunner.utils import INFO
import torch.utils.data as Data

import pickle
import random
import os

"""
    This script define the parent class to deal with some common function for Dataset

    Author: SunnerLi
"""

class BaseDataset(Data.Dataset):
    def __init__(self):
        self.save_file = False
        self.files = None
        self.split_files = None

    def generateIndexList(self, a, size):
        """
            Generate the list of index which will be picked
            This function will be used as train-test-split

            Arg:    a       - The list of images
                    size    - Int, the length of list you want to create
            Ret:    The index list
        """
        result = set()
        while len(result) != size:
            result.add(random.randint(0, len(a) - 1))
        return list(result)

    def loadFromFile(self, file_name, check_type = 'image'):
        """
            Load the root and files information from .pkl record file
            This function will return False if the record file format is invalid

            Arg:    file_name   - The name of record file
                    check_type  - Str. The type of the record file you want to check
            Ret:    If the loading procedure are successful or not
        """
        with open(file_name, 'rb') as f:
            obj = pickle.load(f)
            self.type  = obj['type']
            if self.type == check_type:
                INFO("Load from file: {}".format(file_name))
                self.root  = obj['root']
                self.files = obj['files']
                return True
            else:
                INFO("Record file type: {}\tFail to load...".format(self.type))
                INFO("Form the contain from scratch...")
                return False

    def save(self, remain_file_name, split_ratio, split_file_name = ".split.pkl", save_type = 'image'):
        """
            Save the information into record file

            Arg:    remain_file_name    - The path of record file which store the information of remain data
                    split_ratio         - Float. The proportion to split the data. Usually used to split the testing data
                    split_file_name     - The path of record file which store the information of split data
                    save_type           - Str. The type of the record file you want to save
        """
        if self.save_file:
            if not os.path.exists(remain_file_name):
                with open(remain_file_name, 'wb') as f:
                    pickle.dump({
                        'type': save_type,
                        'root': self.root,
                        'files': self.files
                    }, f)
            if split_ratio:
                INFO("Split the dataset, and save as {}".format(split_file_name))
                with open(split_file_name, 'wb') as f:
                    pickle.dump({
                        'type': save_type,
                        'root': self.root,
                        'files': self.split_files
                    }, f) 
from torchvision_sunner.transforms.base import OP
from torchvision_sunner.utils import INFO
from skimage import transform
import numpy as np
import torch

"""
    This script define some complex operations
    These kind of operations should conduct work iteratively (with inherit OP class)

    Author: SunnerLi
"""

class Resize(OP):
    def __init__(self, output_size):
        """
            Resize the tensor to the desired size
            This function only support for nearest-neighbor interpolation
            Since this mechanism can also deal with categorical data

            Arg:    output_size - The tuple (H, W)
        """
        self.output_size = output_size
        INFO("Applied << %15s >>" % self.__class__.__name__)
        INFO("* Notice: the rank format of input tensor should be 'BHWC'")

    def work(self, tensor):
        """
            Resize the tensor
            If the tensor is not in the range of [-1, 1], we will do the normalization automatically

            Arg:    tensor  - The np.ndarray object. The tensor you want to deal with
            Ret:    The resized tensor
        """
        # Normalize the tensor if needed
        mean, std = -1, -1
        min_v = np.min(tensor)
        max_v = np.max(tensor)
        if not (max_v <= 1 and min_v >= -1):
            mean = 0.5 * max_v + 0.5 * min_v
            std  = 0.5 * max_v - 0.5 * min_v
            # print(max_v, min_v, mean, std)
            tensor = (tensor - mean) / std

        # Work
        tensor = transform.resize(tensor, self.output_size, mode = 'constant', order = 0)

        # De-normalize the tensor
        if mean != -1 and std != -1:
            tensor = tensor * std + mean
        return tensor    

class Normalize(OP):
    def __init__(self, mean = [127.5, 127.5, 127.5], std = [127.5, 127.5, 127.5]):
        """
            Normalize the tensor with given mean and standard deviation
            * Notice: If you didn't give mean and std, the result will locate in [-1, 1]

            Args:
                mean        - The mean of the result tensor
                std         - The standard deviation
        """
        self.mean = mean
        self.std  = std
        INFO("Applied << %15s >>" % self.__class__.__name__)
        INFO("* Notice: the rank format of input tensor should be 'BCHW'")
        INFO("*****************************************************************")
        INFO("* Notice: You should must call 'ToFloat' before normalization")
        INFO("*****************************************************************")
        if self.mean == [127.5, 127.5, 127.5] and self.std == [127.5, 127.5, 127.5]:
            INFO("* Notice: The result will locate in [-1, 1]")

    def work(self, tensor):
        """
            Normalize the tensor

            Arg:    tensor  - The np.ndarray object. The tensor you want to deal with
            Ret:    The normalized tensor
        """
        if tensor.shape[0] != len(self.mean):
            raise Exception("The rank format should be BCHW, but the shape is {}".format(tensor.shape))
        result = []
        for t, m, s in zip(tensor, self.mean, self.std):
            result.append((t - m) / s)
        tensor = np.asarray(result)

        # Check if the normalization can really work
        if np.min(tensor) < -1 or np.max(tensor) > 1:
            raise Exception("Normalize can only work with float tensor",
                "Try to call 'ToFloat()' before normalization")
        return tensor

class UnNormalize(OP):
    def __init__(self, mean = [127.5, 127.5, 127.5], std = [127.5, 127.5, 127.5]):
        """
            Unnormalize the tensor with given mean and standard deviation
            * Notice: If you didn't give mean and std, the function will assume that the original distribution locates in [-1, 1]

            Args:
                mean    - The mean of the result tensor
                std     - The standard deviation
        """
        self.mean = mean
        self.std = std
        INFO("Applied << %15s >>" % self.__class__.__name__)
        INFO("* Notice: the rank format of input tensor should be 'BCHW'")
        if self.mean == [127.5, 127.5, 127.5] and self.std == [127.5, 127.5, 127.5]:
            INFO("* Notice: The function assume that the input range is [-1, 1]")

    def work(self, tensor):
        """
            Un-normalize the tensor

            Arg:    tensor  - The np.ndarray object. The tensor you want to deal with
            Ret:    The un-normalized tensor
        """
        if tensor.shape[0] != len(self.mean):
            raise Exception("The rank format should be BCHW, but the shape is {}".format(tensor.shape))
        result = []
        for t, m, s in zip(tensor, self.mean, self.std):
            result.append(t * s + m)
        tensor = np.asarray(result)
        return tensor

class ToGray(OP):
    def __init__(self):
        """
            Change the tensor as the gray scale
            The function will turn the BCHW tensor into B1HW gray-scaled tensor
        """
        INFO("Applied << %15s >>" % self.__class__.__name__)
        INFO("* Notice: the rank format of input tensor should be 'BCHW'")

    def work(self, tensor):
        """
            Make the tensor into gray-scale

            Arg:    tensor  - The np.ndarray object. The tensor you want to deal with
            Ret:    The gray-scale tensor, and the rank of the tensor is B1HW
        """
        if tensor.shape[0] == 3:
            result = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
            result = np.expand_dims(result, axis = 0)
        elif tensor.shape[0] != 4:
            result = 0.299 * tensor[:, 0] + 0.587 * tensor[:, 1] + 0.114 * tensor[:, 2]
            result = np.expand_dims(result, axis = 1)
        else:
            raise Exception("The rank format should be BCHW, but the shape is {}".format(tensor.shape))
        return result
import numpy as np
import torch

"""
    This class define the parent class of operation

    Author: SunnerLi
"""

class OP():
    """
        The parent class of each operation
        The goal of this class is to adapting with different input format
    """
    def work(self, tensor):
        """
            The virtual function to define the process in child class

            Arg:    tensor  - The np.ndarray object. The tensor you want to deal with
        """
        raise NotImplementedError("You should define your own function in the class!")

    def __call__(self, tensor):
        """
            This function define the proceeding of the operation
            There are different choice toward the tensor parameter
            1. torch.Tensor and rank is CHW
            2. np.ndarray and rank is CHW
            3. torch.Tensor and rank is TCHW
            4. np.ndarray and rank is TCHW

            Arg:    tensor  - The tensor you want to operate
            Ret:    The operated tensor
        """
        isTensor = type(tensor) == torch.Tensor
        if isTensor:
            tensor_type = tensor.type()
            tensor = tensor.cpu().data.numpy()
        if len(tensor.shape) == 3:
            tensor = self.work(tensor)
        elif len(tensor.shape) == 4:
            tensor = np.asarray([self.work(_) for _ in tensor])
        else:
            raise Exception("We dont support the rank format {}".format(tensor.shape),
                "If the rank of the tensor shape is only 2, you can call 'GrayStack()'")
        if isTensor:
            tensor = torch.from_numpy(tensor)
            tensor = tensor.type(tensor_type)
        return tensor
"""
    This script define the wrapper of the Torchvision_sunner.transforms

    Author: SunnerLi
"""
from torchvision_sunner.constant import *
from torchvision_sunner.utils import *

from torchvision_sunner.transforms.base import *
from torchvision_sunner.transforms.simple import * 
from torchvision_sunner.transforms.complex import *
from torchvision_sunner.transforms.categorical import *
from torchvision_sunner.transforms.function import *
from torchvision_sunner.transforms.simple import Transpose
from torchvision_sunner.constant import BCHW2BHWC

from skimage import transform
from skimage import io 
import numpy as np
import torch

"""
    This script define the transform function which can be called directly

    Author: SunnerLi
"""

channel_op = None       # Define the channel op which will be used in 'asImg' function

def asImg(tensor, size = None):
    """
        This function provides fast approach to transfer the image into numpy.ndarray
        This function only accept the output from sigmoid layer or hyperbolic tangent output

        Arg:    tensor  - The torch.Variable object, the rank format is BCHW or BHW
                size    - The tuple object, and the format is (height, width)
        Ret:    The numpy image, the rank format is BHWC
    """
    global channel_op
    result = tensor.detach()

    # 1. Judge the rank first
    if len(tensor.size()) == 3:
        result = torch.stack([result, result, result], 1)

    # 2. Judge the range of tensor (sigmoid output or hyperbolic tangent output)
    min_v = torch.min(result).cpu().data.numpy()
    max_v = torch.max(result).cpu().data.numpy()
    if max_v > 1.0 or min_v < -1.0:
        raise Exception('tensor value out of range...\t range is [' + str(min_v) + ' ~ ' + str(max_v))
    if min_v < 0:
        result = (result + 1) / 2

    # 3. Define the BCHW -> BHWC operation
    if channel_op is None:
        channel_op = Transpose(BCHW2BHWC)

    # 3. Rest               
    result = channel_op(result)
    result = result.cpu().data.numpy()
    if size is not None:
        result_list = []
        for img in result:
            result_list.append(transform.resize(img, (size[0], size[1]), mode = 'constant', order = 0) * 255)
        result = np.stack(result_list, axis = 0)
    else:
        result *= 255.
    result = result.astype(np.uint8)
    return result
from torchvision_sunner.utils import INFO
from torchvision_sunner.constant import *
import numpy as np
import torch

"""
    This script define some operation which are rather simple
    The operation only need to call function once (without inherit OP class)

    Author: SunnerLi
"""

class ToTensor():
    def __init__(self):
        """
            Change the tensor into torch.Tensor type
        """
        INFO("Applied << %15s >>" % self.__class__.__name__)

    def __call__(self, tensor):
        """
            Arg:    tensor - The torch.Tensor or other type. The tensor you want to deal with
        """
        if type(tensor) == np.ndarray:
            tensor = torch.from_numpy(tensor)
        return tensor

class ToFloat():
    def __init__(self):
        """
            Change the tensor into torch.FloatTensor
        """        
        INFO("Applied << %15s >>" % self.__class__.__name__)

    def __call__(self, tensor):
        """
            Arg:    tensor - The torch.Tensor object. The tensor you want to deal with
        """
        return tensor.float()

class Transpose():
    def __init__(self, direction = BHWC2BCHW):
        """
            Transfer the rank of tensor into target one

            Arg:    direction   - The direction you want to do the transpose
        """        
        self.direction = direction
        if self.direction == BHWC2BCHW:
            INFO("Applied << %15s >>, The rank format is BCHW" % self.__class__.__name__)
        elif self.direction == BCHW2BHWC:
            INFO("Applied << %15s >>, The rank format is BHWC" % self.__class__.__name__)
        else:
            raise Exception("Unknown direction symbol: {}".format(self.direction))

    def __call__(self, tensor):
        """
            Arg:    tensor - The torch.Tensor object. The tensor you want to deal with
        """
        if self.direction == BHWC2BCHW:
            tensor = tensor.transpose(-1, -2).transpose(-2, -3)
        else:
            tensor = tensor.transpose(-3, -2).transpose(-2, -1)
        return tensor
from torchvision_sunner.utils import INFO
from torchvision_sunner.constant import *
from torchvision_sunner.transforms.simple import Transpose

from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import pickle
import torch
import json
import os

"""
    This script define the categorical-related operations, including:
        1. getCategoricalMapping
        2. CategoricalTranspose

    Author: SunnerLi
"""

# ----------------------------------------------------------------------------------------
#   Define the IO function toward pallete file
# ----------------------------------------------------------------------------------------

def load_pallete(file_name):
    """
        Load the pallete object from file

        Arg:    file_name   - The name of pallete .json file
        Ret:    The list of pallete object
    """
    # Load the list of dict from files (key is str)
    palletes_str_key = None
    with open(file_name, 'r') as f:
        palletes_str_key = json.load(f)

    # Change the key into color tuple
    palletes = [OrderedDict()] * len(palletes_str_key)
    for folder in range(len(palletes_str_key)):
        for key in palletes_str_key[folder].keys():
            tuple_key = list()
            for v in key.split('_'):
                tuple_key.append(int(v))
            palletes[folder][tuple(tuple_key)] = palletes_str_key[folder][key]
    return palletes

def save_pallete(pallete, file_name):
    """
        Load the pallete object from file

        Arg:    pallete     - The list of OrderDict objects
                file_name   - The name of pallete .json file
    """
    # Change the key into str
    pallete_str_key = [dict()] * len(pallete)
    for folder in range(len(pallete)):
        for key in pallete[folder].keys():

            str_key = '_'.join([str(_) for _ in key])
            pallete_str_key[folder][str_key] = pallete[folder][key]

    # Save into file
    with open(file_name, 'w') as f:
        json.dump(pallete_str_key, f)

# ----------------------------------------------------------------------------------------
#   Define the categorical-related operations
# ----------------------------------------------------------------------------------------

def getCategoricalMapping(loader = None, path = 'torchvision_sunner_categories_pallete.json'):
    """
        This function can statistic the different category with color
        And return the list of the mapping OrderedDict object

        Arg:    loader  - The ImageLoader object
                path    - The path of pallete file
        Ret:    The list of OrderDict object (palletes object)
    """
    INFO("Applied << %15s >>" % getCategoricalMapping.__name__)
    INFO("* Notice: the rank format of input tensor should be 'BHWC'")
    INFO("* Notice: The range of tensor should be in [0, 255]")
    if os.path.exists(path):
        palletes = load_pallete(path)
    else:
        INFO(">> Load from scratch, please wait...")

        # Get the number of folder
        folder_num = 0
        for img_list in loader:
            folder_num = len(img_list)
            break

        # Initialize the pallete list
        palletes = [OrderedDict()] * folder_num
        color_sets = [set()] * folder_num

        # Work
        for img_list in tqdm(loader):
            for folder_idx in range(folder_num):
                img = img_list[folder_idx]
                if torch.max(img) > 255 or torch.min(img) < 0:
                    raise Exception('tensor value out of range...\t range is [' + str(torch.min(img)) + ' ~ ' + str(torch.max(img)))
                img = img.cpu().data.numpy().astype(np.uint8)
                img = np.reshape(img, [-1, 3])
                color_sets[folder_idx] |= set([tuple(_) for _ in img])

        # Merge the color
        for i in range(folder_num):
            for color in color_sets[i]:
                if color not in palletes[i].keys():
                    palletes[i][color] = len(palletes[i])
        save_pallete(palletes, path)

    return palletes

class CategoricalTranspose():
    def __init__(self, pallete = None, direction = COLOR2INDEX, index_default = 0):
        """
            Transform the tensor into the particular format
            We support for 3 different kinds of format:
                1. one hot image
                2. index image
                3. color
            
            Arg:    pallete         - The pallete object (default is None)
                    direction       - The direction you want to change
                    index_default   - The default index if the color cannot be found in the pallete
        """
        self.pallete = pallete
        self.direction = direction
        self.index_default = index_default
        INFO("Applied << %15s >> , direction: %s" % (self.__class__.__name__, self.direction))
        INFO("* Notice: The range of tensor should be in [-1, 1]")
        INFO("* Notice: the rank format of input tensor should be 'BCHW'")

    def fn_color_to_index(self, tensor):
        """
            Transfer the tensor from the RGB colorful format into the index format

            Arg:    tensor  - The tensor obj. The tensor you want to deal with
            Ret:    The tensor with index format
        """
        if self.pallete is None:
            raise Exception("The direction << %s >> need the pallete object" % self.direction)
        tensor = tensor.transpose(-3, -2).transpose(-2, -1).cpu().data.numpy()
        size_tuple = list(np.shape(tensor))
        tensor = (tensor * 127.5 + 127.5).astype(np.uint8)
        tensor = np.reshape(tensor, [-1, 3])
        tensor = [tuple(_) for _ in tensor]
        tensor = [self.pallete.get(_, self.index_default) for _ in tensor]
        tensor = np.asarray(tensor)
        size_tuple[-1] = 1
        tensor = np.reshape(tensor, size_tuple)
        tensor = torch.from_numpy(tensor).transpose(-1, -2).transpose(-2, -3)
        return tensor

    def fn_index_to_one_hot(self, tensor):
        """
            Transfer the tensor from the index format into the one-hot format

            Arg:    tensor  - The tensor obj. The tensor you want to deal with
            Ret:    The tensor with one-hot format
        """
        # Get the number of classes
        tensor = tensor.transpose(-3, -2).transpose(-2, -1)
        size_tuple = list(np.shape(tensor))
        tensor = tensor.view(-1).cpu().data.numpy()
        channel = np.amax(tensor) + 1

        # Get the total number of pixel
        num_of_pixel = 1
        for i in range(len(size_tuple) - 1):
            num_of_pixel *= size_tuple[i]

        # Transfer as ont-hot format
        one_hot_tensor = np.zeros([num_of_pixel, channel])
        for i in range(channel):
            one_hot_tensor[tensor == i, i] = 1

        # Recover to origin rank format and shape
        size_tuple[-1] = channel
        tensor = np.reshape(one_hot_tensor, size_tuple)
        tensor = torch.from_numpy(tensor).transpose(-1, -2).transpose(-2, -3)
        return tensor

    def fn_one_hot_to_index(self, tensor):
        """
            Transfer the tensor from the one-hot format into the index format

            Arg:    tensor  - The tensor obj. The tensor you want to deal with
            Ret:    The tensor with index format
        """
        _, tensor = torch.max(tensor, dim = 1)
        tensor = tensor.unsqueeze(1)
        return tensor

    def fn_index_to_color(self, tensor):
        """
            Transfer the tensor from the index format into the RGB colorful format

            Arg:    tensor  - The tensor obj. The tensor you want to deal with
            Ret:    The tensor with RGB colorful format
        """
        if self.pallete is None:
            raise Exception("The direction << %s >> need the pallete object" % self.direction)
        tensor = tensor.transpose(-3, -2).transpose(-2, -1).cpu().data.numpy()
        reverse_pallete = {self.pallete[x]: x for x in self.pallete}
        batch, height, width, channel = np.shape(tensor)
        tensor = np.reshape(tensor, [-1])
        tensor = np.round(tensor, decimals=0)
        tensor = np.vectorize(reverse_pallete.get)(tensor)
        tensor = np.reshape(np.asarray(tensor).T, [batch, height, width, len(reverse_pallete[0])])
        tensor = torch.from_numpy((tensor - 127.5) / 127.5).transpose(-1, -2).transpose(-2, -3)
        return tensor

    def __call__(self, tensor):
        if self.direction == COLOR2INDEX:
            return self.fn_color_to_index(tensor)
        elif self.direction == INDEX2COLOR:
            return self.fn_index_to_color(tensor)
        elif self.direction == ONEHOT2INDEX:
            return self.fn_one_hot_to_index(tensor)
        elif self.direction == INDEX2ONEHOT:
            return self.fn_index_to_one_hot(tensor)
        elif self.direction == ONEHOT2COLOR:
            return self.fn_index_to_color(self.fn_one_hot_to_index(tensor))
        elif self.direction == COLOR2ONEHOT:
            return self.fn_index_to_one_hot(self.fn_color_to_index(tensor))
        else:
            raise Exception("Unknown direction: {}".format(self.direction))
"""
    This script defines the constant which will be used in Torchvision_sunner package

    Author: SunnerLi
"""

# Constant
UNDER_SAMPLING = 0
OVER_SAMPLING = 1
BCHW2BHWC = 0
BHWC2BCHW = 1

# Categorical constant
ONEHOT2INDEX = 'one_hot_to_index'
INDEX2ONEHOT = 'index_to_one_hot'
ONEHOT2COLOR = 'one_hot_to_color'
COLOR2ONEHOT = 'color_to_one_hot'
INDEX2COLOR  = 'index_to_color'
COLOR2INDEX  = 'color_to_index'

# Verbose flag
verbose = True
# coding: UTF-8
"""
    @author: samuel ko
"""
import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms


from networks_stylegan import StyleGenerator, StyleDiscriminator
from networks_gan import Generator, Discriminator
from utils import plotLossCurve
from loss import gradient_penalty
from opts import TrainOptions

from torchvision.utils import save_image
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.optim as optim
import numpy as np
import torch
import os

# Hyper-parameters
CRITIC_ITER = 5


def main(opts):
    # Create the data loader
    loader = sunnerData.DataLoader(sunnerData.ImageDataset(
        root=[[opts.path]],
        transform=transforms.Compose([
            sunnertransforms.Resize((1024, 1024)),
            sunnertransforms.ToTensor(),
            sunnertransforms.ToFloat(),
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
            sunnertransforms.Normalize(),
        ])),
        batch_size=opts.batch_size,
        shuffle=True,
    )

    # Create the model
    G = StyleGenerator(bs=opts.batch_size).to(opts.device)
    D = StyleDiscriminator().to(opts.device)

    # G = Generator().to(opts.device)
    # D = Discriminator().to(opts.device)

    # Create the criterion, optimizer and scheduler
    optim_D = optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optim_G = optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))
    scheduler_D = optim.lr_scheduler.ExponentialLR(optim_D, gamma=0.99)
    scheduler_G = optim.lr_scheduler.ExponentialLR(optim_G, gamma=0.99)

    # Train
    fix_z = torch.randn([opts.batch_size, 512]).to(opts.device)
    Loss_D_list = [0.0]
    Loss_G_list = [0.0]
    for ep in range(opts.epoch):
        bar = tqdm(loader)
        loss_D_list = []
        loss_G_list = []
        for i, (real_img,) in enumerate(bar):
            # =======================================================================================================
            #   Update discriminator
            # =======================================================================================================
            # Compute adversarial loss toward discriminator
            real_img = real_img.to(opts.device)
            real_logit = D(real_img)
            fake_img = G(torch.randn([real_img.size(0), 512]).to(opts.device))
            fake_logit = D(fake_img.detach())
            d_loss = -(real_logit.mean() - fake_logit.mean()) + gradient_penalty(real_img.data, fake_img.data, D) * 10.0
            loss_D_list.append(d_loss.item())

            # Update discriminator
            optim_D.zero_grad()
            d_loss.backward()
            optim_D.step()

            # =======================================================================================================
            #   Update generator
            # =======================================================================================================
            if i % CRITIC_ITER == 0:
                # Compute adversarial loss toward generator
                fake_img = G(torch.randn([opts.batch_size, 512]).to(opts.device))
                fake_logit = D(fake_img)
                g_loss = -fake_logit.mean()
                loss_G_list.append(g_loss.item())

                # Update generator
                D.zero_grad()
                optim_G.zero_grad()
                g_loss.backward()
                optim_G.step()
            bar.set_description(" {} [G]: {} [D]: {}".format(ep, loss_G_list[-1], loss_D_list[-1]))

        # Save the result
        Loss_G_list.append(np.mean(loss_G_list))
        Loss_D_list.append(np.mean(loss_D_list))
        fake_img = G(fix_z)
        save_image(fake_img, os.path.join(opts.det, 'images', str(ep) + '.png'), nrow=4, normalize=True)
        state = {
            'G': G.state_dict(),
            'D': D.state_dict(),
            'Loss_G': Loss_G_list,
            'Loss_D': Loss_D_list,
        }
        torch.save(state, os.path.join(opts.det, 'models', 'latest.pth'))

        scheduler_D.step()
        scheduler_G.step()

    # Plot the total loss curve
    Loss_D_list = Loss_D_list[1:]
    Loss_G_list = Loss_G_list[1:]
    plotLossCurve(opts, Loss_D_list, Loss_G_list)


if __name__ == '__main__':
    opts = TrainOptions().parse()
    main(opts)
# coding: UTF-8
"""
    @author: samuel ko
"""

from matplotlib import pyplot as plt
import os

def plotLossCurve(opts, Loss_D_list, Loss_G_list):
    plt.figure()
    plt.plot(Loss_D_list, '-')
    plt.title("Loss curve (Discriminator)")
    plt.savefig(os.path.join(opts.det, 'images', 'loss_curve_discriminator.png'))

    plt.figure()
    plt.plot(Loss_G_list, '-o')
    plt.title("Loss curve (Generator)")
    plt.savefig(os.path.join(opts.det, 'images', 'loss_curve_generator.png'))
# coding: UTF-8
"""
    @author: samuel ko
"""

from torch.autograd import Variable
from torch.autograd import grad
import torch.autograd as autograd
import torch.nn as nn
import torch

import numpy as np

def gradient_penalty(x, y, f):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).to(x.device)
    z = x + alpha * (y - x)

    # gradient penalty
    z = Variable(z, requires_grad=True).to(x.device)
    o = f(z)
    g = grad(o, z, grad_outputs=torch.ones(o.size()).to(z.device), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1)**2).mean()
    return gp


def R1Penalty(real_img, f):
    # gradient penalty
    reals = Variable(real_img, requires_grad=True).to(real_img.device)
    real_logit = f(reals)
    apply_loss_scaling = lambda x: x * torch.exp(x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))
    undo_loss_scaling = lambda x: x * torch.exp(-x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))

    real_logit = apply_loss_scaling(torch.sum(real_logit))
    real_grads = grad(real_logit, reals, grad_outputs=torch.ones(real_logit.size()).to(reals.device), create_graph=True)[0].view(reals.size(0), -1)
    real_grads = undo_loss_scaling(real_grads)
    r1_penalty = torch.sum(torch.mul(real_grads, real_grads))
    return r1_penalty


def R2Penalty(fake_img, f):
    # gradient penalty
    fakes = Variable(fake_img, requires_grad=True).to(fake_img.device)
    fake_logit = f(fakes)
    apply_loss_scaling = lambda x: x * torch.exp(x * torch.Tensor([np.float32(np.log(2.0))]).to(fake_img.device))
    undo_loss_scaling = lambda x: x * torch.exp(-x * torch.Tensor([np.float32(np.log(2.0))]).to(fake_img.device))

    fake_logit = apply_loss_scaling(torch.sum(fake_logit))
    fake_grads = grad(fake_logit, fakes, grad_outputs=torch.ones(fake_logit.size()).to(fakes.device), create_graph=True)[0].view(fakes.size(0), -1)
    fake_grads = undo_loss_scaling(fake_grads)
    r2_penalty = torch.sum(torch.mul(fake_grads, fake_grads))
    return r2_penalty

# coding: UTF-8
"""
    @author: samuel ko
"""
import torch.nn.functional as F
import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, z_dims=512, d=64):
        super().__init__()
        self.deconv1 = nn.utils.spectral_norm(nn.ConvTranspose2d(z_dims, d * 8, 4, 1, 0))
        self.deconv2 = nn.utils.spectral_norm(nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1))
        self.deconv3 = nn.utils.spectral_norm(nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1))
        self.deconv4 = nn.utils.spectral_norm(nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1))
        self.deconv5 = nn.utils.spectral_norm(nn.ConvTranspose2d(d * 2, d, 4, 2, 1))
        self.deconv6 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    def forward(self, input):
        input = input.view(input.size(0), input.size(1), 1, 1)  # 1 x 1
        x = F.relu(self.deconv1(input))  # 4 x 4
        x = F.relu(self.deconv2(x))  # 8 x 8
        x = F.relu(self.deconv3(x))  # 16 x 16
        x = F.relu(self.deconv4(x))  # 32 x 32
        x = F.relu(self.deconv5(x))  # 64 x 64
        x = F.tanh(self.deconv6(x))  # 128 x 128
        return x


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.layer1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.layer2 = nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False))
        self.layer3 = nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False))
        self.layer4 = nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False))
        self.layer5 = nn.utils.spectral_norm(nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False))
        self.layer6 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

    def forward(self, input):
        out = F.leaky_relu(self.layer1(input), 0.2, inplace=True)  # 64 x 64
        out = F.leaky_relu(self.layer2(out), 0.2, inplace=True)  # 32 x 32
        out = F.leaky_relu(self.layer3(out), 0.2, inplace=True)  # 16 x 16
        out = F.leaky_relu(self.layer4(out), 0.2, inplace=True)  # 8 x 8
        out = F.leaky_relu(self.layer5(out), 0.2, inplace=True)  # 4 x 4
        out = F.leaky_relu(self.layer6(out), 0.2, inplace=True)  # 1 x 1
        return out.view(-1, 1)
