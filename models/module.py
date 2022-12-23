import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys

from .unsup_homo import *
from .unsup_modules import *

def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
            # x = F.gelu(x)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
            # x = F.gelu(x)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
            # x = F.gelu(x)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
            # x = F.gelu(x)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)
        # return F.gelu(self.bn(self.conv(x)))


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)
        # return F.gelu(self.bn(self.conv(x)))


class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class Hourglass3d(nn.Module):
    def __init__(self, channels):
        super(Hourglass3d, self).__init__()

        self.conv1a = ConvBnReLU3D(channels, channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv1b = ConvBnReLU3D(channels * 2, channels * 2, kernel_size=3, stride=1, pad=1)

        self.conv2a = ConvBnReLU3D(channels * 2, channels * 4, kernel_size=3, stride=2, pad=1)
        self.conv2b = ConvBnReLU3D(channels * 4, channels * 4, kernel_size=3, stride=1, pad=1)

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose3d(channels * 4, channels * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels * 2))

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose3d(channels * 2, channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels))

        self.redir1 = ConvBn3D(channels, channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = ConvBn3D(channels * 2, channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        dconv2 = F.relu(self.dconv2(conv2) + self.redir2(conv1), inplace=True)
        # dconv2 = F.gelu(self.dconv2(conv2) + self.redir2(conv1))
        dconv1 = F.relu(self.dconv1(dconv2) + self.redir1(x), inplace=True)
        # dconv1 = F.gelu(self.dconv1(dconv2) + self.redir1(x))
        return dconv1


def homo_warping_backup(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] or [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / (proj_xyz[:, 2:3, :, :] + 1e-9)  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros', align_corners=True)
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] or [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        invalid = (proj_xyz[:, 2:3, :, :]<1e-6).squeeze(1) # [B, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / (proj_xyz[:, 2:3, :, :] + + 1e-9)  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_x_normalized[invalid] = -99.
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_y_normalized[invalid] = -99.
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros', align_corners=True)
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


class DeConv2dFuse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(DeConv2dFuse, self).__init__()

        self.deconv = Deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                               bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2d(2*out_channels, out_channels, kernel_size, stride=1, padding=1,
                           bn=bn, relu=relu, bn_momentum=bn_momentum)

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x


class FeatureNet(nn.Module):
    def __init__(self, base_channels, num_stage=3, stride=4, arch_mode="unet"):
        super(FeatureNet, self).__init__()
        assert arch_mode in ["unet", "fpn"], print("mode must be in 'unet' or 'fpn', but get:{}".format(arch_mode))
        print("*************feature extraction arch mode:{}****************".format(arch_mode))
        self.arch_mode = arch_mode
        self.stride = stride
        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]

        if self.arch_mode == 'unet':
            if num_stage == 3:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)
                self.deconv2 = DeConv2dFuse(base_channels * 2, base_channels, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
                self.out3 = nn.Conv2d(base_channels, base_channels, 1, bias=False)
                self.out_channels.append(2 * base_channels)
                self.out_channels.append(base_channels)

            elif num_stage == 2:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
                self.out_channels.append(2 * base_channels)
        elif self.arch_mode == "fpn":
            final_chs = base_channels * 4
            if num_stage == 3:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
                self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
                self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels * 2)
                self.out_channels.append(base_channels)

            elif num_stage == 2:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels)

    def forward(self, x):
        """forward.

        :param x: [B, C, H, W]
        :return outputs: stage1 [B, 32, 128, 160], stage2 [B, 16, 256, 320], stage3 [B, 8, 512, 640]
        """
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = {}
        out = self.out1(intra_feat)
        outputs["stage1"] = out
        if self.arch_mode == "unet":
            if self.num_stage == 3:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

                intra_feat = self.deconv2(conv0, intra_feat)
                out = self.out3(intra_feat)
                outputs["stage3"] = out

            elif self.num_stage == 2:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

        elif self.arch_mode == "fpn":
            if self.num_stage == 3:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
                out = self.out3(intra_feat)
                outputs["stage3"] = out

            elif self.num_stage == 2:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

        return outputs




class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(CostRegNet, self).__init__()
        self.conv0 = Conv3d(in_channels, base_channels, padding=1)

        self.conv1 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x

class CostRegNet_cas(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(CostRegNet_cas, self).__init__()
        self.convini = Conv3d(in_channels, base_channels, padding=1)
        self.conv0 = Conv3d(base_channels, base_channels, padding=1)

        self.conv1 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        convini = self.convini(x)
        conv0 = self.conv0(convini)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x

class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        concat = F.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


def depth_wta(p, depth_values):
    wta_index_map = torch.argmax(p, dim=1, keepdim=True).type(torch.long)   # B,1,H,W
    wta_depth_map = torch.gather(depth_values, 1, wta_index_map).squeeze(1)
    return wta_depth_map


def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)
    return depth


def info_entropy_loss(prob_volume, prob_volume_pre, mask):
    # prob_colume should be processed after SoftMax
    B,D,H,W = prob_volume.shape
    LSM = nn.LogSoftmax(dim=1)
    valid_points = torch.sum(mask, dim=[1,2])+1e-6
    entropy = -1*(torch.sum(torch.mul(prob_volume, LSM(prob_volume_pre)), dim=1)).squeeze(1)
    entropy_masked = torch.sum(torch.mul(mask, entropy), dim=[1,2])
    return torch.mean(entropy_masked / valid_points)


def entropy_loss_mask(prob_volume, depth_gt, mask, depth_value, return_prob_map=False):
    # depth_value: B * NUM or [B,D,H,W]
    mask_true = mask
    valid_pixel_num = torch.sum(mask_true, dim=[1, 2]) + 1e-6

    shape = depth_gt.shape  # B,H,W

    depth_num = depth_value.shape[1]
    if len(depth_value.shape) < 3:
        depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2, 3, 0, 1)  # B,N,H,W
    else:
        depth_value_mat = depth_value

    if depth_value.shape[2] == 256:
        mask_in_range = torch.logical_and(depth_value_mat.min(1)[0] - 2.5 < depth_gt,
                                          depth_value_mat.max(1)[0] + 2.5 > depth_gt)
        mask_true = torch.logical_and(mask_true, mask_in_range)
    elif depth_value.shape[2] == 512:
        mask_in_range = torch.logical_and(depth_value_mat.min(1)[0] - 2.5/2 < depth_gt,
                                          depth_value_mat.max(1)[0] + 2.5/2 > depth_gt)
        mask_true = torch.logical_and(mask_true, mask_in_range)

    gt_index_image = torch.argmin(torch.abs(depth_value_mat - depth_gt.unsqueeze(1)), dim=1)

    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1)  # B, 1, H, W

    # gt index map -> gt one hot volume (B x 1 x H x W )
    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1,
                                                                                                           gt_index_image,
                                                                                                           1)

    # cross entropy image (B x D X H x W)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume+ 1e-6), dim=1).squeeze(1)  # B, 1, H, W
    # cross_entropy_image = -torch.sum(gt_index_volume * torch.log_softmax(prob_volume_pre, dim=1), dim=1).squeeze(1)  # B, 1, H, W

    # masked cross entropy loss
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image)  # valid pixel
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])

    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num)  # Origin use sum : aggregate with batch
    # winner-take-all depth map
    wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)

    if return_prob_map:
        photometric_confidence = torch.max(prob_volume, dim=1)[0]  # output shape dimension B * H * W
        return masked_cross_entropy, wta_depth_map, photometric_confidence
    return masked_cross_entropy, wta_depth_map



def focal_loss(prob_volume, depth_gt, mask, depth_value,depth_interval, return_prob_map=False):
    # depth_value: B * NUM or [B,D,H,W]
    mask_true = mask
    valid_pixel_num = torch.sum(mask_true, dim=[1, 2]) + 1e-6

    shape = depth_gt.shape  # B,H,W

    depth_num = depth_value.shape[1]
    if len(depth_value.shape) < 3:
        depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2, 3, 0, 1)  # B,N,H,W
    else:
        depth_value_mat = depth_value

    if depth_value.shape[2] == 256:
        mask_in_range = torch.logical_and(depth_value_mat.min(1)[0] - depth_interval < depth_gt,
                                          depth_value_mat.max(1)[0] + depth_interval > depth_gt)
        mask_true = torch.logical_and(mask_true, mask_in_range)
    elif depth_value.shape[2] == 512:
        mask_in_range = torch.logical_and(depth_value_mat.min(1)[0] - depth_interval/2. < depth_gt,
                                          depth_value_mat.max(1)[0] + depth_interval/2. > depth_gt)
        mask_true = torch.logical_and(mask_true, mask_in_range)

    gt_index_image = torch.argmin(torch.abs(depth_value_mat - depth_gt.unsqueeze(1)), dim=1)

    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1)  # B, 1, H, W

    positive_volume = torch.gather(prob_volume,1,gt_index_image)  # N,1,H,W
    cross_entropy_image = -(torch.clamp(positive_volume.log(), min=-100) * ((1 - positive_volume) ** 2)).squeeze(1)  # B, H, W

    # masked cross entropy loss
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image)  # valid pixel
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])

    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num)  # Origin use sum : aggregate with batch
    # winner-take-all depth map
    wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)

    if return_prob_map:
        photometric_confidence = torch.max(prob_volume, dim=1)[0]  # output shape dimension B * H * W
        return masked_cross_entropy, wta_depth_map, photometric_confidence
    return masked_cross_entropy, wta_depth_map


def entropy_loss(prob_volume, depth_gt, mask, depth_value, return_prob_map=False):
    '''
    Cross entropy loss function. This is also a approximation to the KL loss.
    '''
    # depth_value: B * NUM or [B,D,H,W]
    mask_true = mask
    valid_pixel_num = torch.sum(mask_true, dim=[1,2]) + 1e-6

    shape = depth_gt.shape          # B,H,W

    depth_num = depth_value.shape[1]
    if len(depth_value.shape) < 3:
        depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2,3,0,1)     # B,N,H,W
    else:
        depth_value_mat = depth_value

    gt_index_image = torch.argmin(torch.abs(depth_value_mat-depth_gt.unsqueeze(1)), dim=1)

    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1) # B, 1, H, W

    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1, gt_index_image, 1)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume + 1e-6), dim=1).squeeze(1) # B, 1, H, W
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image) # valid pixel
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])
    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num) # Origin use sum : aggregate with batch
    # winner-take-all depth map
    wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)

    if return_prob_map:
        photometric_confidence = torch.max(prob_volume, dim=1)[0] # output shape dimension B * H * W
        return masked_cross_entropy, wta_depth_map, photometric_confidence
    return masked_cross_entropy, wta_depth_map



def kl_distance(pred_mu, pred_sigma, mu, sigma):
    ''' Calculate the KL distance between two distributionzs.
    :param: pred_mu, extracted attribute vector with shape [B, 1, H, W]
    :param: mu, mean tensor with shape [B, 1, H, W]
    '''
    dis = (0.5 * (torch.log(sigma/pred_sigma) + (pred_sigma + (pred_mu - mu)**2)/sigma - 1.0))#.sum(dim=1).mean() # B,1,H,W
    return dis


def kl_distance_sp(pred_mu, pred_sigma, mu, sigma):
    ''' Calculate the KL distance between two distributionzs.
    :param: pred_mu, extracted attribute vector with shape [B, 1, H, W]
    :param: mus, mean tensor with shape [B, 1, H, W]
    '''
    kl_loss = 0.0
    for i, pred_mu in enumerate(pred_mu):
        kl_loss += (0.5 * (torch.log(sigma/pred_sigma[i].exp()) + (pred_sigma[i].exp() + (pred_mu - mu[:,i:i+1])**2)/sigma - 1.0))#.sum(dim=1).mean()
    return kl_loss


def get_mu_sigma(prob_volume, depth_values):
    ''' Calculate the mu and sigma from probability volume and depth values.
    :param: prob_volume [B, D, H, W]
    :param: depth_values [B, D, H, W]
    '''
    # shape = prob_volume.shape
    # if len(depth_values.shape) < 3: # [B, D]
    #     depth_values = depth_values.repeat(shape[2], shape[3], 1, 1).permute(2,3,0,1)     # B,N,H,W
    mu = torch.sum(prob_volume * depth_values, 1, keepdim=True)     # B,1,H,W
    sigma = torch.sqrt(torch.sum((depth_values - mu)**2 * prob_volume, 1))  #B,1,H,W
    return mu, sigma


def kl_loss(prob_volume, gt_mu, gt_sigma, mask, depth_value):
    ''' Kullback Leibler divergence based loss function to mesure the distance between
    the student model's predicted probability and the pseudo probability distribution.
    '''
    # depth_value: B * NUM or [B,D,H,W]
    mask_true = mask
    valid_pixel_num = torch.sum(mask_true, dim=[1,2]) + 1e-6

    shape = gt_mu.shape          # B,1,H,W
    depth_num = depth_value.shape[1]

    if len(depth_value.shape) < 3:
        depth_value_mat = depth_value.repeat(shape[2], shape[3], 1, 1).permute(2,3,0,1)     # B,N,H,W
    else:
        depth_value_mat = depth_value

    # compute the mu and sigma of predicted prob_volume and depth values
    pred_mu, pred_sigma = get_mu_sigma(prob_volume, depth_value_mat)    # both are: B,1,H,W
    kl_loss_img = kl_distance(pred_mu+1e-6, pred_sigma+1e-6, gt_mu+1e-6, gt_sigma+1e-6)# / float(depth_num)    # B,1,H,W

    masked_kl_loss_image = torch.mul(mask_true, kl_loss_img) # valid pixel
    masked_kl_loss_image = torch.sum(masked_kl_loss_image, dim=[1, 2])
    kl_loss_value = torch.mean(masked_kl_loss_image / valid_pixel_num)

    return kl_loss_value


def  approx_kl_loss(prob_volume, mu_gt, mask, depth_value):
    ''' Approximation to the KL loss (Gaussion distribution with extreme low sigma (dwon to zero)).
    '''
    mask_true = mask
    valid_pixel_num = torch.sum(mask_true, dim=[1,2]) + 1e-6
    shape = mu_gt.shape          # B,H,W
    depth_num = depth_value.shape[1]
    if len(depth_value.shape) < 3:
        depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2,3,0,1)     # B,N,H,W
    else:
        depth_value_mat = depth_value

    gt_index_image = torch.argmin(torch.abs(depth_value_mat-mu_gt.unsqueeze(1)), dim=1)
    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1) # B, 1, H, W
    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1, gt_index_image, 1)
    approx_kl_image = -torch.sum(gt_index_volume * torch.log(prob_volume + 1e-6), dim=1).squeeze(1) # B, 1, H, W
    masked_approx_kl_image = torch.mul(mask_true, approx_kl_image) # valid pixel
    masked_approx_kl = torch.sum(masked_approx_kl_image, dim=[1, 2])
    masked_approx_kl = torch.mean(masked_approx_kl / valid_pixel_num)
    wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)

    return masked_approx_kl, wta_depth_map


def cas_mvsnet_loss_kl(inputs, depth_gt_ms, sigma_gt_ms, mask_ms, **kwargs):

    depth_loss_weights = kwargs.get("dlossw", None)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    total_approx_kl =  torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    total_kl =  torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        # depth_est = stage_inputs["depth"]
        prob_volume = stage_inputs["prob_volume"]
        depth_values = stage_inputs["depth_values"]
        gt_mu = depth_gt_ms[stage_key]   # B,H,W
        gt_sigma = sigma_gt_ms[stage_key]   # B,H,W
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        kl_weight = 0.5 #5.0
        approx_weight = 2.0

        # compute the kl loss
        kl_loss_value = kl_loss(prob_volume, gt_mu.unsqueeze(1), gt_sigma.unsqueeze(1), mask, depth_values)
        kl_loss_value = kl_loss_value * kl_weight
        total_kl += kl_loss_value

        # compute the approx kl loss
        approx_kl_value, depth_wta = approx_kl_loss(prob_volume, gt_mu, mask, depth_values)
        approx_kl_value = approx_kl_value * approx_weight
        total_approx_kl += approx_kl_value

        depth_loss = F.smooth_l1_loss(depth_wta[mask], gt_mu[mask], reduction='mean')

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * kl_loss_value
            total_loss += depth_loss_weights[stage_idx] * approx_kl_value
        else:
            total_loss += kl_loss_value
            total_loss += approx_kl_value

    return total_loss, depth_loss, total_kl*5.0, total_approx_kl, depth_wta


def cas_mvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    total_entropy =  torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"]
        # Added to modify info-entropy-loss
        prob_volume = stage_inputs["prob_volume"]
        depth_values = stage_inputs["depth_values"]
        # prob_volume_pre = stage_inputs["prob_volume_pre"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        entropy_weight = 2.0

        entro_loss, depth_entropy = entropy_loss(prob_volume, depth_gt, mask, depth_values)
        entro_loss = entro_loss * entropy_weight
        # depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
        depth_loss = F.smooth_l1_loss(depth_entropy[mask], depth_gt[mask], reduction='mean')
        total_entropy += entro_loss

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            # total_loss += depth_loss_weights[stage_idx] * depth_loss
            total_loss += depth_loss_weights[stage_idx] * entro_loss
        else:
            # total_loss += 1.0 * depth_loss
            total_loss += entro_loss

    return total_loss, depth_loss, entro_loss, depth_entropy


def cas_mvsnet_loss_bld(inputs, depth_gt_ms, mask_ms, depth_interval, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=True)
    total_entropy =  torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"]
        prob_volume = stage_inputs["prob_volume"]
        depth_values = stage_inputs["depth_values"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        entropy_weight = 2.0

        entro_loss, depth_entropy = focal_loss(prob_volume, depth_gt, mask, depth_values, depth_interval)
        entro_loss = entro_loss * entropy_weight
        depth_loss = F.smooth_l1_loss(depth_entropy[mask], depth_gt[mask], reduction='mean')
        total_entropy += entro_loss

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            # total_loss = total_loss + depth_loss_weights[stage_idx] * depth_loss
            total_loss = total_loss +  depth_loss_weights[stage_idx] * entro_loss
        else:
            # total_loss = total_loss + 1.0 * depth_loss
            total_loss = total_loss +  entro_loss
    abs_err = (depth_gt_ms['stage3'] - inputs["stage3"]["depth"]).abs()
    # print(f"abs_err shape in loss func : {abs_err.shape}")
    # print(f"depth_interval shape in loss func : {depth_interval.shape}")
    abs_err_scaled = abs_err / (depth_interval * 192. / 128.)#.unsqueeze(1).unsqueeze(2)
    # abs_err_scaled = abs_err / depth_interval
    mask = mask_ms["stage3"]
    mask = mask > 0.5
    epe = abs_err_scaled[mask].mean()
    less1 = (abs_err_scaled[mask] < 1.).to(depth_gt_ms['stage3'].dtype).mean()
    less3 = (abs_err_scaled[mask] < 3.).to(depth_gt_ms['stage3'].dtype).mean()

    return total_loss, depth_loss, epe, less1, less3


class unsup_loss(nn.Module):
    def __init__(self):
        super(unsup_loss, self).__init__()
        self.ssim = SSIM()

    def forward(self, inputs, imgs, sample_cams, num_views=5, **kwargs):
        # def unsup_loss(inputs, imgs, sample_cams, num_views=5, **kwargs):

        depth_loss_weights = kwargs.get("dlossw", None)

        total_loss = torch.tensor(0.0, dtype=torch.float32, device=inputs['stage1']["depth"].device, requires_grad=False)
        total_photo_loss = torch.tensor(0.0, dtype=torch.float32, device=inputs['stage1']["depth"].device, requires_grad=False)
        total_feature_loss = torch.tensor(0.0, dtype=torch.float32, device=inputs['stage1']["depth"].device, requires_grad=False)

        reconstr_loss = torch.tensor(0.0, dtype=torch.float32, device=inputs['stage1']["depth"].device, requires_grad=False)
        ssim_loss = torch.tensor(0.0, dtype=torch.float32, device=inputs['stage1']["depth"].device, requires_grad=False)
        smooth_loss = torch.tensor(0.0, dtype=torch.float32, device=inputs['stage1']["depth"].device, requires_grad=False)

        for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
            depth_est = stage_inputs["depth"].unsqueeze(1)   # b,1,h,w
            features = stage_inputs['features']

            log_var = stage_inputs['var']   # b,h,w

            ref_img = imgs[:,0] # b,c,h,w
            scale = depth_est.shape[-1] / ref_img.shape[-1]
            ref_img = F.interpolate(ref_img, scale_factor=scale, mode='bilinear', align_corners=True)
            ref_img = ref_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
            ref_cam = sample_cams[stage_key][:,0]   # b,2,4,4

            ref_feature = features[0].detach()  # b,c,h,w
            ref_feature = ref_feature.permute(0, 2, 3, 1)

            warped_img_list = []
            warped_feature_list = []
            feature_mask_list = []
            mask_list = []
            reprojection_losses = []
            fea_reprojection_losses = []

            for view in range(1, num_views):
                view_img = imgs[:,view]
                view_feature = features[view].detach()
                view_feature = view_feature.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
                view_cam = sample_cams[stage_key][:,view]
                view_img = F.interpolate(view_img, scale_factor=scale, mode='bilinear', align_corners=True)
                view_img = view_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
                # warp view_img to the ref_img using the dmap of the ref_img
                warped_img, mask = inverse_warping(view_img, ref_cam, view_cam, depth_est)
                warped_img_list.append(warped_img)
                mask_list.append(mask)

                warped_fea, fea_mask = inverse_warping(view_feature, ref_cam, view_cam, depth_est)
                warped_feature_list.append(warped_fea)
                feature_mask_list.append(fea_mask)

                reconstr_loss = compute_reconstr_loss(warped_img, ref_img, mask, simple=False)
                fea_reconstr_loss = compute_reconstr_loss(warped_fea, ref_feature, fea_mask, simple=False)
                valid_mask = 1 - mask  # replace all 0 values with INF
                fea_valid_mask = 1 - fea_mask
                reprojection_losses.append(reconstr_loss + 1e4 * valid_mask)
                fea_reprojection_losses.append(fea_reconstr_loss + 1e4 * fea_valid_mask)

                # SSIM loss##
                if view < 3:
                    ssim_loss += torch.mean(self.ssim(ref_img, warped_img, mask))
            del features, view_feature, ref_feature

            ##smooth loss##
            smooth_loss += depth_smoothness(depth_est.unsqueeze(dim=-1), ref_img, 1.0)
            # top-k operates along the last dimension, so swap the axes accordingly
            reprojection_volume = torch.stack(reprojection_losses).permute(1,2,3,4,0)  # [4, 128, 160, 1, 6]
            top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=3, sorted=False)
            top_vals = torch.neg(top_vals)
            top_mask = top_vals < (1e4 * torch.ones_like(top_vals, device=device))
            top_mask = top_mask.float()
            top_vals = torch.mul(top_vals, top_mask)  # [4, 128, 160, 1, 3]
            top_vals = torch.sum(top_vals, dim=-1)  # [4, 128, 160, 1]
            top_vals = top_vals.permute(0, 3, 1, 2)  # [4, 1, 128, 160]

            fea_reprojection_volume = torch.stack(fea_reprojection_losses).permute(1,2,3,4,0)  # [4, 128, 160, 1, 6]
            fea_top_vals, fea_top_inds = torch.topk(torch.neg(fea_reprojection_volume), k=3, sorted=False)
            fea_top_vals = torch.neg(fea_top_vals)
            fea_top_mask = fea_top_vals < (1e4 * torch.ones_like(fea_top_vals, device=device))
            fea_top_mask = fea_top_mask.float()
            fea_top_vals = torch.mul(fea_top_vals, fea_top_mask)  # [4, 128, 160, 1, 3]
            fea_top_vals = torch.sum(fea_top_vals, dim=-1)  # [4, 128, 160, 1]
            fea_top_vals = fea_top_vals.permute(0, 3, 1, 2)  # [4, 1, 128, 160]

            loss1 = torch.mean(torch.exp(-log_var) * top_vals)
            loss2 = torch.mean(log_var)
            loss3 = torch.mean(torch.exp(-log_var) * fea_top_vals)
            reconstr_loss = 0.5 * (loss1 + 0.25*loss3 + 0.1 * loss2)

            # self.reconstr_loss = torch.mean()
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += (12 * 2 * reconstr_loss + 6 * ssim_loss + 0.18 * smooth_loss) * depth_loss_weights[stage_idx]
            total_photo_loss += 12*loss1* depth_loss_weights[stage_idx]
            total_feature_loss += 3*loss3* depth_loss_weights[stage_idx]
            # self.unsup_loss = (0.8 * self.reconstr_loss + 0.2 * self.ssim_loss + 0.067 * self.smooth_loss) * 15

        return total_loss, 12*2*reconstr_loss, 6*ssim_loss, \
                0.18*smooth_loss, total_photo_loss, total_feature_loss


def get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth=192.0, min_depth=0.0):
    #shape, (B, H, W)
    #cur_depth: (B, H, W)
    #return depth_range_values: (B, D, H, W)
    cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel)  # (B, H, W)
    cur_depth_max = (cur_depth + ndepth / 2 * depth_inteval_pixel)
    # cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel).clamp(min=0.0)   #(B, H, W)
    # cur_depth_max = (cur_depth_min + (ndepth - 1) * depth_inteval_pixel).clamp(max=max_depth)

    assert cur_depth.shape == torch.Size(shape), "cur_depth:{}, input shape:{}".format(cur_depth.shape, shape)
    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=cur_depth.device,
                                                                  dtype=cur_depth.dtype,
                                                                  requires_grad=False).reshape(1, -1, 1,
                                                                                               1) * new_interval.unsqueeze(1))

    return depth_range_samples


def get_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, device, dtype, shape,
                           max_depth=192.0, min_depth=0.0, use_inverse_depth=False):

    if cur_depth.dim() == 2:
        cur_depth_min = cur_depth[:, 0]  # (B,)
        cur_depth_max = cur_depth[:, -1]

        if use_inverse_depth is False:
            new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, )  Shouldn't cal this if we use inverse depth
            depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=device, dtype=dtype,
                                                                        requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)) #(B, D)
            depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[1], shape[2]) #(B, D, H, W)
        else:
            # When use inverse_depth for T&T
            depth_range_samples = cur_depth.repeat(1, 1, shape[1], shape[2]) #(B, D, H, W)

    else:
        depth_range_samples = get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth, min_depth)

    return depth_range_samples


