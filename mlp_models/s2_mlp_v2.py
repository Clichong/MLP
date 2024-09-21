import torch
import torch.nn as nn
import einops
from timm.models.layers import trunc_normal_, to_2tuple

class SpatialShiftBlock1(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, w, h, c = x.size()
        x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
        x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
        x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
        x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
        return x


class SpatialShiftBlock2(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, w, h, c = x.size()
        x[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
        x[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
        x[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
        x[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
        return x


class SplitAttention(nn.Module):

    def __init__(self, hidden_dim, k=3):
        super().__init__()
        self.k = k
        self.mlp1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim * k, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        b, k, h, w, c = x.shape     # 8, 3, 14, 14, 512
        x = x.reshape(b, k, -1, c)                  # bs,k,n,c
        a = torch.sum(torch.sum(x, 1), 1)           # bs,c
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))  # bs,kc
        hat_a = hat_a.reshape(b, self.k, c)         # bs,k,c
        bar_a = self.softmax(hat_a)                 # bs,k,c
        attention = bar_a.unsqueeze(-2)             # #bs,k,1,c
        out = attention * x                         # #bs,k,n,c
        out = torch.sum(out, 1).reshape(b, h, w, c)
        return out


class SpatialShift(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp1 = nn.Linear(hidden_dim, hidden_dim * 3)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.spatialshift_block1 = SpatialShiftBlock1()
        self.spatialshift_block2 = SpatialShiftBlock2()
        self.split_attention = SplitAttention(hidden_dim)

    def forward(self, x):
        b, w, h, c = x.size()
        x = self.mlp1(x)
        x1 = self.spatialshift_block1(x[:, :, :, :c])
        x2 = self.spatialshift_block2(x[:, :, :, c:c*2])
        x3 = x[:, :, :, c*2:]
        x_all = torch.stack([x1, x2, x3], 1)
        out = self.split_attention(x_all)   # 8, 3, 14, 14, 512
        out = self.mlp2(out)
        return out


class S2MLPv2Block(nn.Module):

    def __init__(self, hidden_dim, expansion_factor):
        super().__init__()
        # self.ln = nn.LayerNorm(hidden_dim)
        # self.model_p1 = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.GELU(),
        #     SpatialShift(hidden_dim),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        # )
        self.model_p1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            SpatialShift(hidden_dim),
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


class DownSample(nn.Module):

    def __init__(self, in_c, out_c, patch_size):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=to_2tuple(patch_size), stride=(2, 2), padding=1)

    def forward(self, x):
        out = einops.rearrange(x, 'b w h c -> b c w h')
        out = self.conv(out)
        out = einops.rearrange(out, 'b c w h -> b w h c')
        return out


class S2MLPv2(nn.Module):

    def __init__(self, patch_size, image_size, in_c, hidden_dim, expansion_factor, num_blocks, num_classes):
        super().__init__()
        self.num_blocks = num_blocks
        self.token_dim = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.patch_embed = nn.Conv2d(in_c, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.stages = nn.Sequential(
            *[nn.Sequential(
                S2MLPv2Block(hidden_dim, expansion_factor)
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
        out = einops.rearrange(out, 'b h w c -> b c h w')
        out = einops.reduce(out, 'b c h w -> b c', 'mean')
        out = self.head(out)
        return out


if __name__ == '__main__':
    image_size = 224
    x = torch.rand(8, 3, image_size, image_size)
    model = S2MLPv2((16, 16), (image_size, image_size), 3, 224, 4, 8, 5)
    # model = S2MLPv2((8, 8), (32, 32), 3, 32, 4, 2, 10)
    print(model)

    output = model(x)
    print(output.shape)

