import torch
import torch.nn as nn
import einops
from timm.models.layers import trunc_normal_, to_2tuple

class SpatialShiftBlock(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, w, h, c = x.size()
        x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
        x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
        x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
        x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
        return x


class S2MLPBlock(nn.Module):

    def __init__(self, hidden_dim, expansion_factor):
        super().__init__()
        # self.ln = nn.LayerNorm(hidden_dim)
        self.model_p1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            SpatialShiftBlock(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.model_p2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
        )

    def forward(self, x):
        out = self.model_p1(x)
        x = out + x
        out = self.model_p2(x)
        return out + x


class S2MLPv1(nn.Module):

    def __init__(self, patch_size, image_size, in_c, hidden_dim, expansion_factor, num_blocks, num_classes):
        super().__init__()
        self.num_blocks = num_blocks
        self.token_dim = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.patch_embed = nn.Conv2d(in_c, hidden_dim, kernel_size=patch_size, stride=patch_size)
        # self.s2mlp_block = S2MLPBlock(hidden_dim, expansion_factor)
        self.stages = nn.Sequential(
            *[nn.Sequential(
                S2MLPBlock(hidden_dim, expansion_factor)
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
        out = einops.rearrange(out, 'b c h w -> b h w c')
        out = self.stages(out)
        out = einops.rearrange(out, 'b h w c -> b c h w')
        out = einops.reduce(out, 'b c h w -> b c', 'mean')
        out = self.head(out)
        return out


if __name__ == '__main__':
    image_size = 32
    x = torch.rand(8, 3, image_size, image_size)
    model = S2MLPv1((16, 16), (image_size, image_size), 3, 224, 4, 6, 5)
    # model = S2MLPv1((8, 8), (image_size, image_size), 3, 32, 4, 6, 10)
    print(model)

    output = model(x)
    print(output.shape)


