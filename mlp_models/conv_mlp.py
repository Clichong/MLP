import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, to_2tuple

class MLP(nn.Module):

    def __init__(self, hidden_dim, expansion_factor):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * expansion_factor)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim * expansion_factor, hidden_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.fc2(out)
        return out


class ConvTokenizer(nn.Module):
    def __init__(self, embedding_dim=64):
        super(ConvTokenizer, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3,
                      embedding_dim // 2,
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim // 2,
                      embedding_dim // 2,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim // 2,
                      embedding_dim,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3),
                         stride=(2, 2),
                         padding=(1, 1),
                         dilation=(1, 1))
        )

    def forward(self, x):
        return self.block(x)


class ConvStage(nn.Module):
    def __init__(self,
                 num_blocks=2,
                 embedding_dim_in=64,
                 hidden_dim=128,
                 embedding_dim_out=224):
        super(ConvStage, self).__init__()
        self.conv_blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = nn.Sequential(
                nn.Conv2d(embedding_dim_in, hidden_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, embedding_dim_in, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                nn.BatchNorm2d(embedding_dim_in),
                nn.ReLU(inplace=True)
            )
            self.conv_blocks.append(block)
        self.downsample = nn.Conv2d(embedding_dim_in,
                                 embedding_dim_out,
                                 kernel_size=(3, 3),
                                 stride=(2, 2),
                                 padding=(1, 1))

    def forward(self, x):
        for block in self.conv_blocks:
            x = x + block(x)
        return self.downsample(x)


class ConvMLPBlock(nn.Module):

    def __init__(self, hidden_dim, expansion_factor):
        super().__init__()
        self.conv_mlp_1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            MLP(hidden_dim, expansion_factor)
        )
        self.de_conv = nn.Conv2d(hidden_dim, hidden_dim,
                                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                 groups=hidden_dim, bias=False)
        self.de_conv_norm = nn.LayerNorm(hidden_dim)
        self.conv_mlp_2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            MLP(hidden_dim, expansion_factor)
        )

    def forward(self, x):
        out = self.conv_mlp_1(x)
        x = out + x
        out = self.de_conv(self.de_conv_norm(x).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = self.conv_mlp_2(x)
        return out + x


class ConvMLP(nn.Module):

    def __init__(self, patch_size, image_size, in_c, hidden_dim, expansion_factor, num_blocks, num_classes, Conv_Embedding=False):
        super().__init__()
        self.num_blocks = num_blocks
        self.Conv_Embedding = Conv_Embedding
        self.token_size = (image_size[0] // patch_size[0],  image_size[1] // patch_size[1])
        self.token_dim = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        if Conv_Embedding:
            self.patch_embed = ConvTokenizer()
            self.conv_stages = ConvStage()
        else:
            self.patch_embed = nn.Conv2d(in_c, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.stages = nn.Sequential(
            *[nn.Sequential(
                ConvMLPBlock(hidden_dim, expansion_factor)
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
        if self.Conv_Embedding:
            out = self.patch_embed(x)
            out = self.conv_stages(out)     # 8, 224, 28, 28
        else:
            out = self.patch_embed(x)       # 8, 224, 14, 14
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
    # model = ConvMLP((16, 16), (224, 224), 3, 224, 4, 6, 5)
    model = ConvMLP((8, 8), (32, 32), 3, 32, 4, 2, 10)
    print(model)

    output = model(x)
    print(output.shape)