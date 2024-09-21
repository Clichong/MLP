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
        self.fc2 = nn.Linear(hidden_dim * expansion_factor, hidden_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.fc2(out)
        return out


class SparseModule(nn.Module):

    def __init__(self, hidden_dim, token_size):
        super().__init__()
        self.proj_h = nn.Linear(token_size[0], token_size[0])
        self.proj_w = nn.Linear(token_size[1], token_size[1])
        self.fuse = nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x_h = self.proj_h(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)    # b w h c -> b w c h -> b w h c
        x_w = self.proj_w(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)    # b w h c -> b c h w -> b w h c
        x_c = x

        x_cat = torch.cat([x_h, x_w, x_c], dim=-1).permute(0, 3, 1, 2)  # b w h c -> b c w h
        out = self.fuse(x_cat).permute(0, 2, 3, 1)      # b c w h -> b w h c

        return out


class SparseMLPBlock(nn.Module):

    def __init__(self, hidden_dim, expansion_factor, token_size):
        super().__init__()
        self.model_p1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            SparseModule(hidden_dim, token_size),
        )
        self.model_p2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            MLP(hidden_dim, expansion_factor)
        )

    def forward(self, x):
        out = self.model_p1(x)
        x = out + x
        out = self.model_p2(x)
        return out + x


class SparseMLP(nn.Module):

    def __init__(self, patch_size, image_size, in_c, hidden_dim, expansion_factor, num_blocks, num_classes):
        super().__init__()
        self.num_blocks = num_blocks
        self.token_size = (image_size[0] // patch_size[0],  image_size[1] // patch_size[1])
        self.token_dim = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.patch_embed = nn.Conv2d(in_c, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.stages = nn.Sequential(
            *[nn.Sequential(
                SparseMLPBlock(hidden_dim, expansion_factor, self.token_size)
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
    x = torch.rand(8, 3, 32, 32)
    model = SparseMLP((16, 16), (224, 224), 3, 224, 4, 1, 5)
    # model = SparseMLP((8, 8), (32, 32), 3, 32, 4, 2, 10)
    print(model)

    output = model(x)
    print(output.shape)