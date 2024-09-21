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


class ASMLP(nn.Module):

    def __init__(self, patch_size, image_size, in_c, hidden_dim, expansion_factor, num_blocks, num_classes):
        super().__init__()
        self.num_blocks = num_blocks
        self.token_dim = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.patch_embed = nn.Conv2d(in_c, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.stages = nn.Sequential(
            *[nn.Sequential(
                ASMLPBlock(hidden_dim, expansion_factor)
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
        out = einops.reduce(out, 'b c w h -> b c', 'mean')
        out = self.head(out)
        return out


if __name__ == '__main__':
    image_size = 224
    patch_size = 8
    x = torch.rand(8, 3, image_size, image_size)
    # model = ASMLP((16, 16), (image_size, image_size), 3, 224, 4, 6, 5)
    model = ASMLP(to_2tuple(patch_size), to_2tuple(image_size), 3, 32, 4, 2, 10)
    print(model)

    output = model(x)
    print(output.shape)