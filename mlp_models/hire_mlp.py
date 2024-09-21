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


class HireModuleMLP(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim_out),
        )

    def forward(self, x):
        out = self.fc(x)
        return out


class CrossRegion(nn.Module):
    def __init__(self, step=1, dim=1):
        super().__init__()
        self.step = step
        self.dim = dim

    def forward(self, x):
        return torch.roll(x, self.step, self.dim)


class HireModule(nn.Module):

    def __init__(self, hidden_dim, h=2, w=2, cross_region_step=1):
        super().__init__()

        # x: b w h c  --> dim_h = 2 / dim_w = 1
        self.inner_region_h = Rearrange('b w (h group) c -> b w group (h c)', h=h)
        self.inner_region_w = Rearrange('b (w group) h c -> b group h (w c)', w=w)
        self.inner_region_restore_h = Rearrange('b w group (h c) -> b w (h group) c', h=h)
        self.inner_region_restore_w = Rearrange('b group h (w c) -> b (w group) h c', w=w)

        self.cross_region_h = CrossRegion(cross_region_step, dim=2)
        self.cross_region_w = CrossRegion(cross_region_step, dim=1)
        self.cross_region_restore_h = CrossRegion(-cross_region_step, dim=2)
        self.cross_region_restore_w = CrossRegion(-cross_region_step, dim=1)

        self.fc_h = HireModuleMLP(h * hidden_dim, hidden_dim // 2, h * hidden_dim)
        self.fc_w = HireModuleMLP(w * hidden_dim, hidden_dim // 2, w * hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):

        x_h = self.inner_region_h(self.cross_region_h(x))
        x_w = self.inner_region_w(self.cross_region_w(x))
        x_c = x

        x_h = self.fc_h(x_h)
        x_w = self.fc_w(x_w)
        x_c = self.fc_c(x_c)

        x_h = self.cross_region_restore_h(self.inner_region_restore_h(x_h))
        x_w = self.cross_region_restore_w(self.inner_region_restore_w(x_w))

        out = x_h + x_w + x_c
        return out


class HireMLPBlock(nn.Module):

    def __init__(self, hidden_dim, expansion_factor):
        super().__init__()
        self.model_p1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            HireModule(hidden_dim),
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


class HireMLP(nn.Module):

    def __init__(self, patch_size, image_size, in_c, hidden_dim, expansion_factor, num_blocks, num_classes):
        super().__init__()
        self.num_blocks = num_blocks
        self.token_dim = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.patch_embed = nn.Conv2d(in_c, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.stages = nn.Sequential(
            *[nn.Sequential(
                HireMLPBlock(hidden_dim, expansion_factor)
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
    # model = HireMLP((16, 16), (224, 224), 3, 224, 4, 1, 5)
    model = HireMLP((8, 8), (32, 32), 3, 32, 4, 2, 10)
    print(model)

    output = model(x)
    print(output.shape)