import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math

act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
tanh = nn.Tanh()

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.vgg19_54(img)


class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, nf=64, gc=32, res_scale=0.2, bias=True):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.act = act

        # def block(in_features, out_features, non_linearity=True):
        #     layers = [nn.Conv2d(in_features, out_features, 3, 1, 1, bias=True)]
        #     if non_linearity:
        #         layers += [nn.LeakyReLU()]
        #         # layers += [nn.GELU()]
        #     return nn.Sequential(*layers)
        #
        # self.b1 = block(in_features=1 * nf)
        # self.b2 = block(in_features=2 * nf)
        # self.b3 = block(in_features=3 * nf)
        # self.b4 = block(in_features=4 * nf)
        # self.b5 = block(in_features=5 * nf, non_linearity=False)
        # self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.act(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.act(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * self.res_scale + x
        # inputs = x
        # for block in self.blocks:
        #     out = block(inputs)
        #     inputs = torch.cat([inputs, out], 1)
        # return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, nf, gc=32, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.RDB1 = DenseResidualBlock(nf, gc)
        self.RDB2 = DenseResidualBlock(nf, gc)
        self.RDB3 = DenseResidualBlock(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x
        # return self.dense_blocks(x).mul(self.res_scale) + x


class GeneratorRRDB(nn.Module):
    def __init__(self, channels=3, nf=64, num_res_blocks=23, gc=32):
        super(GeneratorRRDB, self).__init__()
        # First layer
        self.conv1 = nn.Conv2d(channels, nf, kernel_size=3, stride=1, padding=1, bias=True)
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(nf=nf, gc=gc) for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)

        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            act,
            nn.Conv2d(nf, channels, kernel_size=3, stride=1, padding=1)
        )

        # Final activation
        self.fact = tanh

        self.act = act
        self.tanh = tanh

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.conv3(out)
        out = self.fact(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 16), int(in_width / 16)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_nf, out_nf, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_nf, out_nf, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_nf))
            layers.append(act)
            layers.append(nn.Conv2d(out_nf, out_nf, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_nf))
            layers.append(act)
            return layers

        layers = []
        in_nf = in_channels
        for i, out_nf in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_nf, out_nf, first_block=(i == 0)))
            in_nf = out_nf

        layers.append(nn.Conv2d(out_nf, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
