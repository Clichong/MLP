import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, to_2tuple

class MLP(nn.Module):

    def __init__(self, hidden_dim, expansion_factor, activation=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * expansion_factor)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_dim * expansion_factor, hidden_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        return out


class GlobalPerceptron(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.average_pool = Reduce('n h w c -> n c', 'mean')
        # self.fc = MLP(hidden_dim, expansion_factor=1, activation=nn.ReLU)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 4, hidden_dim)

    def forward(self, x):
        n, h, w, c = x.size()
        out = self.average_pool(x)
        out = self.fc2(self.relu(self.fc1(out))).view(n, c, 1, 1)
        out = F.sigmoid(out)
        return out      # torch.Size([8, 224, 1, 1])


class LocalPerceptron(nn.Module):

    def __init__(self, num_sharesets, reparam_conv_k, deploy):
        super().__init__()
        self.deploy = deploy
        self.reparam_conv_k = reparam_conv_k
        if not deploy and reparam_conv_k is not None:
            for k in reparam_conv_k:
                conv_branch = self.conv_bn(num_sharesets, num_sharesets, kernel_size=k, stride=1, padding=k // 2, groups=num_sharesets)
                self.__setattr__('repconv{}'.format(k), conv_branch)

    def forward(self, x):
        conv_inputs = x
        if self.reparam_conv_k is not None and not self.deploy:
            conv_out = 0
            for k in self.reparam_conv_k:
                conv_branch = self.__getattr__('repconv{}'.format(k))
                conv_out += conv_branch(conv_inputs)
        return conv_out     # torch.Size([448, 4, 14, 14])

    def conv_bn(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        result = nn.Sequential()
        result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=kernel_size, stride=stride,
                                            padding=padding, groups=groups, bias=False))
        result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
        return result


class ChannelPerceptron(nn.Module):
    def __init__(self, hidden_dim, token_size, num_sharesets):
        super().__init__()
        self.fc3 = nn.Conv2d(token_size[0]*token_size[1]*num_sharesets, token_size[0]*token_size[1]*num_sharesets,
                             kernel_size=1, stride=1, padding=0,
                             bias=False, groups=num_sharesets)
        self.fc3_bn = nn.BatchNorm2d(num_sharesets)

    def forward(self, x):
        nc_s, s, h, w = x.size()
        fc_inputs = x.reshape(nc_s, s*h*w, 1, 1)
        fc_out = self.fc3(fc_inputs).reshape(-1, s, h, w)
        fc_out = self.fc3_bn(fc_out)
        return fc_out     # torch.Size([448, 4, 14, 14])


class RepMLPNetUnit(nn.Module):

    def __init__(self, hidden_dim, token_size, num_sharesets=4, reparam_conv_k=(1, 3), deploy=False):
        super().__init__()
        self.deploy = deploy
        self.num_sharesets = num_sharesets
        self.reparam_conv_k = reparam_conv_k
        self.global_perceptron = GlobalPerceptron(hidden_dim)
        self.local_perceptron = LocalPerceptron(num_sharesets, reparam_conv_k, deploy)
        self.channel_perceptron = ChannelPerceptron(hidden_dim, token_size, num_sharesets)

    def forward(self, x):
        n, h, w, c = x.size()
        input = einops.rearrange(x, 'n h w c -> n c h w')
        input = input.reshape(n*c//self.num_sharesets, self.num_sharesets, h, w)
        x_global = self.global_perceptron(x)
        x_local = self.local_perceptron(input)
        x_channel = self.channel_perceptron(input)
        output = (x_channel + x_local).reshape(n, c, h, w) * x_global
        output = einops.rearrange(output, 'n c h w -> n h w c')
        return output


class RepMLPBlock(nn.Module):

    def __init__(self, hidden_dim, token_size, expansion_factor):
        super().__init__()
        self.repmlp_block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            RepMLPNetUnit(hidden_dim, token_size),
        )
        self.ffn_block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            MLP(hidden_dim, expansion_factor)
        )

    def forward(self, x):
        out = self.repmlp_block(x)
        x = out + x
        out = self.ffn_block(x)
        return out + x


class RepMLPNet(nn.Module):

    def __init__(self, patch_size, image_size, in_c, hidden_dim, expansion_factor, num_blocks, num_classes):
        super().__init__()
        self.num_blocks = num_blocks
        self.token_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.token_dim = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.patch_embed = nn.Conv2d(in_c, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.stages = nn.Sequential(
            *[nn.Sequential(
                RepMLPBlock(hidden_dim, self.token_size, expansion_factor)
            ) for i in range(self.num_blocks)]
        )
        self.head = nn.Linear(hidden_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.patch_embed(x)
        out = einops.rearrange(out, 'b c w h -> b w h c')
        out = self.stages(out)
        out = einops.rearrange(out, 'b w h c -> b c w h')
        out = einops.reduce(out, 'b c h w -> b c', 'mean')
        out = self.head(out)
        return out


if __name__ == '__main__':
    image_size = 224
    patch_size = 8
    x = torch.rand(8, 3, image_size, image_size)
    model = RepMLPNet((16, 16), (224, 224), 3, 224, 4, 1, 5)
    # model = RepMLPNet((8, 8), (32, 32), 3, 32, 4, 2, 10)
    print(model)

    output = model(x)
    print(output.shape)