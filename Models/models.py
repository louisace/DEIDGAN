import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.spectral_norm import *
import numpy as np
import torchvision
import math
import os

# Generator
class SPADEGenerator(nn.Module):
    @staticmethod
    def modify_commandline_options(parser):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self):
        super().__init__()
        nf = 64
        self.ngf = 64
        self.z_dim = 256
        self.num_upsampling_layers = 'normal'
        self.use_vae = True
        self.crop_size = 128
        self.semantic_nc = 7

        self.sw, self.sh = self.compute_latent_vector_size(self.num_upsampling_layers)

        if self.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(self.z_dim, 8 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.semantic_nc, 8 * nf, 3, padding=1)

        self.head = SPADEResnetBlock(8 * nf, 8 * nf)
        self.adainResBlk_1 = AdainResBlk(8 * nf, 8 * nf)

        self.G_middle = SPADEResnetBlock(8 * nf, 8 * nf)
        self.adainResBlk_2 = AdainResBlk(8 * nf, 8 * nf)

        self.up_0 = SPADEResnetBlock(8 * nf, 4 * nf)
        self.adainResBlk_3 = AdainResBlk(4 * nf, 4 * nf)

        self.up_1 = SPADEResnetBlock(4 * nf, 2 * nf)
        self.adainResBlk_4 = AdainResBlk(2 * nf, 2 * nf)

        self.up_2 = SPADEResnetBlock(2 * nf, nf)
        self.adainResBlk_5 = AdainResBlk(nf, nf)

        final_nc = nf

        if self.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2)
            final_nc = nf // 2

        self.conv_bg1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv_bg2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv_img1 = nn.Conv2d(final_nc, 64, 3, padding=1)
        self.conv_img1_1 = nn.Conv2d(final_nc+64, 64, 3, padding=1)
        self.conv_img2 = nn.Conv2d(64, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

        self.AdaIN_cov1 = nn.Conv2d(3, nf, 4, 2, 1)             # 64 * 64 * 64
        self.AdaIN_cov2 = nn.Conv2d(nf, 2 * nf, 4, 2, 1)        # 128 * 32 * 32
        self.AdaIN_cov3 = nn.Conv2d(2 * nf, 4 * nf, 4, 2, 1)    # 256 * 16 * 16
        self.AdaIN_cov4 = nn.Conv2d(4 * nf, 8 * nf, 4, 2, 1)    # 512 * 8 * 8
        self.AdaIN_cov5 = nn.Conv2d(8 * nf, 16 * nf, 4, 2, 1)   # 1024 * 4 * 4
        self.AdaIN_cov6 = nn.Conv2d(16 * nf, 16 * nf, 4, 2, 1)  # 1024 * 2 * 2
        self.AdaIN_fc1 = nn.Linear(1024 * 2 * 2, 64)
        self.AdaIN_fc2 = nn.Linear(64, 64)

    def compute_latent_vector_size(self, num_upsampling_layers):
        if num_upsampling_layers == 'normal':
            num_up_layers = 5

        sw = self.crop_size // (2**num_up_layers)
        sh = round(sw / 1.0)

        return sw, sh


    def forward(self, input, image, bg, z=None):
        seg = input

        if self.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.z_dim, dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 8 * self.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)
        style_fea1 = F.relu(self.AdaIN_cov1(image))      # B*64*64*64
        style_fea2 = F.relu(self.AdaIN_cov2(style_fea1)) # B*128*32*32
        style_fea3 = F.relu(self.AdaIN_cov3(style_fea2)) # B*256*16*16
        style_fea4 = F.relu(self.AdaIN_cov4(style_fea3)) # B*512*8*8
        style_fea5 = F.relu(self.AdaIN_cov5(style_fea4)) # B*1024*4*4
        style_fea6 = F.relu(self.AdaIN_cov6(style_fea5)) # B*1024*2*2
        style_fea = style_fea6.view(-1, 1024*2*2)
        style_fea = self.AdaIN_fc1(style_fea)
        style_fea = self.AdaIN_fc2(style_fea)

        x = self.head(x, seg)                 # B*1024*4*4
        x = self.adainResBlk_1(x, style_fea)  # B*1024*8*8
        # x = self.up(x)

        x = self.G_middle(x, seg)             # B*1024*8*8
        x = self.adainResBlk_2(x, style_fea)  # B*1024*16*16
        # x = self.up(x)

        x = self.up_0(x, seg)                 # B*512*16*16
        x = self.adainResBlk_3(x, style_fea)  # B*512*32*32
        # x = self.up(x)

        x = self.up_1(x, seg)                 # B*256*32*32
        x = self.adainResBlk_4(x, style_fea)  # B*256*64*64
        # x = self.up(x)

        x = self.up_2(x, seg)                 # B*128*64*64
        x = self.adainResBlk_5(x, style_fea)  # B*128*128*128
        # x = self.up(x)

        bg = F.relu(self.conv_bg1(bg))
        bg = F.relu(self.conv_bg2(bg))
        x1 = torch.cat([x, bg], 1)

        # x1 = F.relu(self.conv_img1(x))
        x1 = F.relu(self.conv_img1_1(x1))
        x1 = self.conv_img2(x1)
        x1 = F.tanh(x1)
        return x1

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        x = self.norm(x)
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * x + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=1, actv=nn.LeakyReLU(0.2), upsample=True):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
    
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
        
    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def forward(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        if self.w_hpf == 0:
            x = (x + self._shortcut(x)) / math.sqrt(2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size))
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        self.norm_G = 'spectralspadesyncbatch3x3'
        self.semantic_nc = 7
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in self.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = self.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, self.semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, self.semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, self.semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx1 = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx2 = self.conv_1(self.actvn(self.norm_1(dx1, seg)))

        out = x_s + dx2

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        param_free_norm_type = 'batch'
        self.ks = 3
        self.norm_nc = norm_nc
        self.label_nc = label_nc

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        # elif param_free_norm_type == 'syncbatch':
        #     self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        self.nhidden = 128

        self.pw = self.ks // 2

    # def mlp(self, nhidden):
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(self.label_nc, self.nhidden, kernel_size=self.ks, padding=self.pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(self.nhidden, self.norm_nc, kernel_size=self.ks, padding=self.pw)
        self.mlp_beta = nn.Conv2d(self.nhidden, self.norm_nc, kernel_size=self.ks, padding=self.pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap.clone())
        gamma = self.mlp_gamma(actv.clone())
        beta = self.mlp_beta(actv.clone())

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

# Discriminator
class MultiscaleDiscriminator(nn.Module):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--netD_subarch', type=str, default='n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        return parser

    def __init__(self):
        super().__init__()
        self.num_D = 2
        self.no_ganFeat_loss = False
        self.n_layers_D = 4

        for i in range(self.num_D):
            subnetD = self.create_single_discriminator()
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self):
        subarch = 'n_layer'
        if subarch == 'n_layer':
            netD = NLayerDiscriminator()
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        result = []
        get_intermediate_features = not self.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def modify_commandline_options(parser):
        parser.add_argument('--n_layers_D', type=int, default=4, help='# layers in each discriminator')
        return parser

    def __init__(self):
        super().__init__()
        self.norm_D = 'spectralinstance'
        self.n_layers_D = 4
        self.no_ganFeat_loss = False
        self.input_nc = 10

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = 64

        norm_layer = get_nonspade_norm_layer(self.norm_D)
        sequence = [[nn.Conv2d(self.input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, self.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == self.n_layers_D - 1 else 2
            sequence = sequence + [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)]]

        sequence = sequence + [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]

class MultiscaleDiscriminator2(nn.Module):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--netD_subarch', type=str, default='n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        return parser

    def __init__(self):
        super().__init__()
        self.num_D = 2
        self.no_ganFeat_loss = False
        self.n_layers_D = 4

        for i in range(self.num_D):
            subnetD = self.create_single_discriminator()
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self):
        subarch = 'n_layer'
        if subarch == 'n_layer':
            netD = NLayerDiscriminator2()
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        result = []
        get_intermediate_features = not self.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator2(nn.Module):
    def modify_commandline_options(parser):
        parser.add_argument('--n_layers_D', type=int, default=4, help='# layers in each discriminator')
        return parser

    def __init__(self):
        super().__init__()
        self.norm_D = 'spectralinstance'
        self.n_layers_D = 4
        self.no_ganFeat_loss = False
        self.input_nc = 6

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = 64

        norm_layer = get_nonspade_norm_layer(self.norm_D)
        sequence = [[nn.Conv2d(self.input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, self.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == self.n_layers_D - 1 else 2
            sequence = sequence + [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)]]

        sequence = sequence + [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]


def get_nonspade_norm_layer(norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        # elif subnorm_type == 'sync_batch':
        #     norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


# VGG
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        print(self.slice1)
        print(self.slice2)
        print(self.slice3)
        print(self.slice4)
        print(self.slice5)

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


# Loss
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss = loss + new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.vgg = torch.nn.DataParallel(self.vgg, device_ids=[0, 1])
        self.criterion = nn.MSELoss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss_style = 0
        for i in range(len(x_vgg)):
            b, c, h, w = x_vgg[i].size()
            for j in range(b):
                x_vgg_style = x_vgg[i][j].view(c, h * w)
                y_vgg_style = y_vgg[i][j].view(c, h * w)
                x_vgg_style = torch.mm(x_vgg_style, x_vgg_style.t())
                y_vgg_style = torch.mm(y_vgg_style, y_vgg_style.t())
                loss_style = loss_style + self.weights[i] * torch.mean((x_vgg_style - y_vgg_style) ** 2) / (c * h * w)
        return loss_style

