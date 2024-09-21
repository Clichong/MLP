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

class SpinModule_s3(nn.Module):

    def __init__(self, hidden_dim, group=2, weightattn=True):
        super().__init__()
        self.group = group
        self.weightattn = weightattn
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        # self.split_attention = SplitAttention(hidden_dim)

    def forward(self, x):
        p1, p2, p3 = x.chunk(3, dim=-1)
        p1 = torch.rot90(p1, k=1, dims=[1, 2])
        p2 = torch.rot90(p2, k=2, dims=[1, 2])
        p3 = torch.rot90(p3, k=3, dims=[1, 2])
        p_all = torch.cat([p1, p2, p3], dim=-1)
        p_all = self.proj(p_all)
        output = p_all + x
        return output


class SpinModule_s2(nn.Module):

    def __init__(self, hidden_dim, group=2, weightattn=True):
        super().__init__()
        self.group = group
        self.weightattn = weightattn
        self.proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj_conv = nn.Conv2d(hidden_dim, hidden_dim * 3, kernel_size=1, stride=1, groups=hidden_dim)
        self.split_attention = SplitAttention(hidden_dim)
        # self.proj_c = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        p1, p2, p3, p4 = x.chunk(4, dim=-1)
        p1 = torch.rot90(p1, k=1, dims=[1, 2])
        p2 = torch.rot90(p2, k=2, dims=[1, 2])
        p3 = torch.rot90(p3, k=3, dims=[1, 2])
        p_all = torch.cat([p1, p2, p3, p4], dim=-1)
        return p_all


class SpinModule_s1(nn.Module):

    def __init__(self, hidden_dim, group=2, weightattn=True):
        super().__init__()
        self.group = group
        self.weightattn = weightattn
        self.proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj_conv = nn.Conv2d(hidden_dim, hidden_dim * 3, kernel_size=1, stride=1, groups=hidden_dim)
        self.split_attention = SplitAttention(hidden_dim)
        # self.proj_c = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        p1, p2, p3 = x.chunk(3, dim=-1)
        p1 = torch.rot90(p1, k=1, dims=[1, 2])
        p2 = torch.rot90(p2, k=3, dims=[1, 2])
        p_all = torch.cat([p1, p2, p3], dim=-1)
        return p_all


class SpinModule(nn.Module):

    def __init__(self, hidden_dim, group=2, weightattn=True):
        super().__init__()
        self.spinflag = True
        self.group = group
        self.weightattn = weightattn
        self.proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj_conv = nn.Conv2d(hidden_dim, hidden_dim * 3, kernel_size=1, stride=1, groups=hidden_dim)
        self.split_attention = SplitAttention(hidden_dim)
        # self.proj_c = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        part = self.proj(x)
        # part = self.proj_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        p1, p2, p3 = part.chunk(3, dim=-1)
        # 默认开启旋转特征图操作
        if self.spinflag:
            p1 = self.spin(p1, group=self.group)
            p2 = self.spin(p2, group=self.group, direction=True)
        # 选择直接相加或者使用split attention相加
        if self.weightattn:
            p_all = torch.stack([p1, p2, p3], dim=1)    # 8, 3, 14, 14, 224
            p_all = self.split_attention(p_all)
        else:
            p_all = p1 + p2 + p3
        # out = self.proj_c(p_all)
        return p_all

    def spin(self, input, group, direction=False):
        if not direction:
            flipflag = 0
        else:
            flipflag = 2
        part = input.chunk(group, dim=-1)
        newpart = []
        for index, p in enumerate(part):
            if index%2:
                p = torch.rot90(p, 1+flipflag, [1, 2])
                newpart.append(p)
            else:
                p = torch.rot90(p, 2, [1, 2])
                newpart.append(p)
        output = torch.cat(newpart, dim=-1)
        return output

class SpinMLPBlock(nn.Module):

    def __init__(self, hidden_dim, expansion_factor, weightattn):
        super().__init__()
        self.model_p1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            SpinModule(hidden_dim, weightattn=weightattn),
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


class SpinMLP(nn.Module):

    def __init__(self, patch_size, image_size, in_c, hidden_dim, expansion_factor, num_blocks, num_classes, weightattn=True):
        super().__init__()
        self.num_blocks = num_blocks
        self.token_dim = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.patch_embed = nn.Conv2d(in_c, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.stages = nn.Sequential(
            *[nn.Sequential(
                SpinMLPBlock(hidden_dim, expansion_factor, weightattn)
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
    model = SpinMLP((4, 4), (224, 224), 3, 224, 4, 4, 5, False)
    # model = SpinMLP((8, 8), (32, 32), 3, 32, 4, 2, 10)
    print(model)

    output = model(x)
    print(output.shape)
