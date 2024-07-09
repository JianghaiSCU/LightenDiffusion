import torch
import torch.nn as nn
import warnings
import os
import math
import torch.nn.functional as F
from einops import rearrange

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_block, self).__init__()

        sequence = []

        sequence += [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        ]

        self.model = nn.Sequential(*sequence)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        out = self.model(x) + self.conv(x)

        return out


class upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsampling, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                                  output_padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.relu(self.conv(x))
        return out


class channel_down(nn.Module):
    def __init__(self, channels):
        super(channel_down, self).__init__()

        self.conv0 = nn.Conv2d(channels * 4, channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1 = nn.Conv2d(channels * 2, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(channels, 3, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = torch.sigmoid(self.conv2(self.relu(self.conv1(self.relu(self.conv0(x))))))

        return out


class channel_up(nn.Module):
    def __init__(self, channels):
        super(channel_up, self).__init__()

        self.conv0 = nn.Conv2d(3, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1 = nn.Conv2d(channels, channels * 2, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(channels * 2, channels * 4, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.conv2(self.relu(self.conv1(self.relu(self.conv0(x)))))

        return out


class feature_pyramid(nn.Module):
    def __init__(self, channels):
        super(feature_pyramid, self).__init__()

        self.convs = nn.Sequential(nn.Conv2d(3, channels, kernel_size=(5, 5), stride=(1, 1), padding=2),
                                   nn.Conv2d(channels, channels, kernel_size=(5, 5), stride=(1, 1), padding=2))

        self.block0 = Res_block(channels, channels)

        self.down0 = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.block1 = Res_block(channels, channels * 2)

        self.down1 = nn.Conv2d(channels * 2, channels * 2, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.block2 = Res_block(channels * 2, channels * 4)

        self.down2 = nn.Conv2d(channels * 4, channels * 4, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):

        level0 = self.down0(self.block0(self.convs(x)))
        level1 = self.down1(self.block1(level0))
        level2 = self.down2(self.block2(level1))

        return level0, level1, level2


class ReconNet(nn.Module):
    def __init__(self, channels):
        super(ReconNet, self).__init__()

        self.pyramid = feature_pyramid(channels)

        self.channel_down = channel_down(channels)
        self.channel_up = channel_up(channels)

        self.block_up0 = Res_block(channels * 4, channels * 4)
        self.block_up1 = Res_block(channels * 4, channels * 4)
        self.up_sampling0 = upsampling(channels * 4, channels * 2)
        self.block_up2 = Res_block(channels * 2, channels * 2)
        self.block_up3 = Res_block(channels * 2, channels * 2)
        self.up_sampling1 = upsampling(channels * 2, channels)
        self.block_up4 = Res_block(channels, channels)
        self.block_up5 = Res_block(channels, channels)
        self.up_sampling2 = upsampling(channels, channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3 = nn.Conv2d(channels, 3, kernel_size=(1, 1), stride=(1, 1), padding=0)

        self.relu = nn.LeakyReLU()

    def forward(self, x, pred_fea=None):

        if pred_fea is None:
            low_fea_down2, low_fea_down4, low_fea_down8 = self.pyramid(x[:, :3, ...])
            low_fea_down8 = self.channel_down(low_fea_down8)

            high_fea_down2, high_fea_down4, high_fea_down8 = self.pyramid(x[:, 3:, ...])
            high_fea_down8 = self.channel_down(high_fea_down8)

            return low_fea_down8, high_fea_down8
        else:
            # =================low ori decoder=================
            low_fea_down2, low_fea_down4, low_fea_down8 = self.pyramid(x[:, :3, ...])

            pred_fea = self.channel_up(pred_fea)

            pred_fea_up2 = self.up_sampling0(
                self.block_up1(self.block_up0(pred_fea) + low_fea_down8))
            pred_fea_up4 = self.up_sampling1(
                self.block_up3(self.block_up2(pred_fea_up2) + low_fea_down4))
            pred_fea_up8 = self.up_sampling2(
                self.block_up5(self.block_up4(pred_fea_up4) + low_fea_down2))

            pred_img = self.conv3(self.relu(self.conv2(pred_fea_up8)))

            return pred_img


class Self_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Self_Attention, self).__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=(1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=(3, 3), stride=(1, 1),
                                    padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super(Cross_Attention, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (dim, num_heads)
            )
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)

        self.query = Depth_conv(in_ch=dim, out_ch=dim)
        self.key = Depth_conv(in_ch=dim, out_ch=dim)
        self.value = Depth_conv(in_ch=dim, out_ch=dim)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        '''
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.attention_head_size,
        )
        print(new_x_shape)
        x = x.view(*new_x_shape)
        '''
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, ctx):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(ctx)
        mixed_value_layer = self.value(ctx)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        ctx_layer = torch.matmul(attention_probs, value_layer)
        ctx_layer = ctx_layer.permute(0, 2, 1, 3).contiguous()

        return ctx_layer


class Retinex_decom(nn.Module):
    def __init__(self, channels):
        super(Retinex_decom, self).__init__()

        self.conv0 = nn.Conv2d(3, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.blocks0 = nn.Sequential(Res_block(channels, channels),
                                     Res_block(channels, channels))

        self.conv1 = nn.Conv2d(1, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.blocks1 = nn.Sequential(Res_block(channels, channels),
                                     Res_block(channels, channels))

        self.cross_attention = Cross_Attention(dim=channels, num_heads=8)
        self.self_attention = Self_Attention(dim=channels, num_heads=8, bias=True)

        self.conv0_1 = nn.Sequential(Res_block(channels, channels),
                                     nn.Conv2d(channels, 3, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.conv1_1 = nn.Sequential(Res_block(channels, channels),
                                     nn.Conv2d(channels, 1, kernel_size=(3, 3), stride=(1, 1), padding=1))

    def forward(self, x):
        init_illumination = torch.max(x, dim=1, keepdim=True)[0]
        init_reflectance = x / init_illumination

        Reflectance, Illumination = (self.blocks0(self.conv0(init_reflectance)),
                                     self.blocks1(self.conv1(init_illumination)))

        Reflectance_final = self.cross_attention(Illumination, Reflectance)

        Illumination_content = self.self_attention(Illumination)

        Reflectance_final = self.conv0_1(Reflectance_final + Illumination_content)
        Illumination_final = self.conv1_1(Illumination - Illumination_content)

        R = torch.sigmoid(Reflectance_final)
        L = torch.sigmoid(Illumination_final)
        L = torch.cat([L for i in range(3)], dim=1)

        return R, L


class CTDN(nn.Module):
    def __init__(self, channels=64):
        super(CTDN, self).__init__()

        self.ReconNet = ReconNet(channels)
        self.retinex = Retinex_decom(channels)

    def forward(self, images, pred_fea=None):

        output = {}
        # =================decomposition low=================
        if pred_fea is None:
            low_fea_down8, high_fea_down8 = self.ReconNet(images, pred_fea=None)

            low_R, low_L = self.retinex(low_fea_down8)
            high_R, high_L = self.retinex(high_fea_down8)

            output["low_R"] = low_R
            output["low_L"] = low_L
            output["low_fea"] = low_fea_down8
            output["high_R"] = high_R
            output["high_L"] = high_L
            output["high_fea"] = high_fea_down8

        else:
            pred_img = self.ReconNet(images[:, :3, ...], pred_fea=pred_fea)
            output["pred_img"] = pred_img

        return output
