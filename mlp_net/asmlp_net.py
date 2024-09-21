import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
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

# def Shift(x, shift_size, dim):
#     pad = shift_size // 2
#     x = F.pad(x, (pad, pad, pad, pad) , "constant", 0)
#     xs = torch.chunk(x, shift_size, 1)
#     x_shift = [torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-pad, pad+1))]
#     x_cat = torch.cat(x_shift, 1)
#     return x_cat[:, :, pad:-pad, pad:-pad]

class Shift(nn.Module):

    def __init__(self, shift_size, dim):
        super().__init__()
        self.dim = dim
        self.shift_size = shift_size
        self.pad = shift_size // 2

    def forward(self, x):
        # pad = shift_size // 2
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(x, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, self.dim) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        return x_cat[:, :, self.pad:-self.pad, self.pad:-self.pad]


class AxialShiftBlock(nn.Module):

    def __init__(self, hidden_dim, shift_size=3):
        super().__init__()
        self.porj_c = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.vertical_shift = nn.Sequential(
            Shift(shift_size, 2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        self.horizontal_shift = nn.Sequential(
            Shift(shift_size, 3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        out = self.porj_c(x)
        x_lr = self.vertical_shift(out)
        x_td = self.horizontal_shift(out)
        out = x_lr + x_td
        out = self.proj(out)
        return out


class ASMLPBlock(nn.Module):

    def __init__(self, hidden_dim, expansion_factor):
        super().__init__()
        self.model_p1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            AxialShiftBlock(hidden_dim),
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


class DownSample(nn.Module):

    def __init__(self, in_c, out_c, patch_size):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=to_2tuple(patch_size), stride=(2, 2), padding=1)

    def forward(self, x):
        out = einops.rearrange(x, 'b w h c -> b c w h')
        out = self.conv(out)
        out = einops.rearrange(out, 'b c w h -> b w h c')
        return out


class ASMLP(nn.Module):

    def __init__(self,
                 patch_size=[4, 3, 3, 3],
                 image_size=(224, 224),
                 in_c=3,
                 hidden_dim=[96, 192, 384, 768],
                 expansion_factor=[4, 4, 4],
                 num_blocks=[2, 4, 2],
                 num_classes=101):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_stages = len(num_blocks)
        self.patch_embed = nn.Conv2d(in_c, hidden_dim[0],
                                     kernel_size=to_2tuple(patch_size[0]), stride=to_2tuple(patch_size[0]))

        down_stage = []
        for i in range(self.num_stages):
            mlp_stage = nn.Sequential(
                *[ASMLPBlock(hidden_dim[i], expansion_factor[i])
                  for _ in range(self.num_blocks[i])]
            )
            down_stage.append(DownSample(hidden_dim[i], hidden_dim[i + 1], patch_size[i + 1]))
            self.__setattr__('stage{}'.format(i), mlp_stage)
        self.down_stage = nn.ModuleList(down_stage)
        self.head = nn.Linear(hidden_dim[-1], num_classes)
        self._init_weights()

    def forward(self, x):
        out = self.patch_embed(x)
        out = einops.rearrange(out, 'b c w h -> b w h c')
        for i in range(self.num_stages):
            mlp_stage = self.__getattr__('stage{}'.format(i))
            down_stage = self.down_stage[i]
            out = mlp_stage(out)
            out = down_stage(out)
        out = einops.rearrange(out, 'b w h c -> b c w h')
        out = einops.reduce(out, 'b c h w -> b c', 'mean')
        out = self.head(out)
        return out

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


def ASMLPNet(*args):
    return ASMLP(*args)


if __name__ == '__main__':
    image_size = 224
    patch_size = 8
    x = torch.rand(8, 3, image_size, image_size)

    model = ASMLPNet()
    print(model)

    output = model(x)
    print(output.shape)