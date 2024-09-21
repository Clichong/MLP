import torch
import torch.nn as nn
import einops
from timm.models.layers import trunc_normal_

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


class PermutatorBlock(nn.Module):

    def __init__(self, hidden_dim, segments, weighted=True):
        super().__init__()
        self.s = segments
        self.weighted = weighted
        self.split_attention = SplitAttention(hidden_dim)
        self.proj_h = nn.Linear(hidden_dim, hidden_dim)
        self.proj_w = nn.Linear(hidden_dim, hidden_dim)
        self.proj_c = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    # H: height, W: width, C: channel S: number of segments
    # x: input tensor of shape (H,W,C)
    def forward(self, x):
        x_h = einops.rearrange(x, 'b h w (c s) -> b w c (h s)', s=self.s)
        x_h = self.proj_h(x_h)
        x_h = einops.rearrange(x_h, 'b w c (h s) -> b h w (c s)', s=self.s)
        x_w = einops.rearrange(x, 'b h w (c s) -> b h c (w s)', s=self.s)
        x_w = self.proj_h(x_w)
        x_w = einops.rearrange(x_w, 'b h c (w s) -> b h w (c s)', s=self.s)
        x_c = self.proj_c(x)
        if self.weighted:
            x  = torch.stack([x_h, x_w, x_c], 1)   # 8, 3, 14, 14, 224
            x = self.split_attention(x)            # 8, 14, 14, 224
        else:
            x = x_h + x_w + x_c
            x = self.proj(x)
        return x


class ViPBlock(nn.Module):

    def __init__(self, hidden_dim, segments, expansion_factor, weighted=True):
        super().__init__()
        self.model_p1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            PermutatorBlock(hidden_dim, segments, weighted=True),
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


class ViP(nn.Module):
    # weighted: 控制是否采用split attention模块
    def __init__(self, patch_size, image_size, in_c, hidden_dim, segments, expansion_factor, num_blocks, num_classes,
                 weighted=True):
        super().__init__()
        self.num_blocks = num_blocks
        self.token_dim = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.patch_embed = nn.Conv2d(in_c, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.stages = nn.Sequential(
            *[nn.Sequential(
                ViPBlock(hidden_dim, segments, expansion_factor, weighted=True)
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
    x = torch.rand(8, 3, image_size, image_size)
    # model = ViP((16, 16), (image_size, image_size), 3, 224, 16, 4, 1, 5)
    model = ViP((8, 8), (image_size, image_size), 3, 32, 8, 4, 1, 10)
    print(model)

    output = model(x)
    print(output.shape)