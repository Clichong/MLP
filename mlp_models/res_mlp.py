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


class Aff(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones([1, 1, hidden_dim]))   # 1
        self.beta = nn.Parameter(torch.zeros([1, 1, hidden_dim]))   # 0

    def forward(self, x):
        x = x * self.alpha + self.beta
        return x


class ResMLPBlock(nn.Module):

    def __init__(self, hidden_dim, token_dim, expansion_factor):
        super().__init__()
        init_values = 0.1
        self.pre_affine_trans = Aff(hidden_dim)
        self.token_mixing = nn.Conv1d(token_dim, token_dim, kernel_size=1)
        self.A = nn.Parameter(init_values * torch.ones((hidden_dim)), requires_grad=True)

        self.post_affine_trans = Aff(hidden_dim)
        self.channel_mixing = MLP(hidden_dim, expansion_factor)
        self.B = nn.Parameter(init_values * torch.ones((hidden_dim)), requires_grad=True)

    def forward(self, x):
        x = self.pre_affine_trans(x)                # like LayerNorm
        z = x + self.A * self.token_mixing(x)       # Token Mixing Operation
        z = self.post_affine_trans(z)               # like LayerNorm
        y = z + self.B * self.channel_mixing(z)     # Channel Mixing Operation
        return y


class ResMLP(nn.Module):

    def __init__(self, patch_size, image_size, in_c, hidden_dim, expansion_factor, num_blocks, num_classes):
        super().__init__()
        self.num_blocks = num_blocks
        self.token_size = (image_size[0] // patch_size[0],  image_size[1] // patch_size[1])
        self.token_dim = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.patch_embed = nn.Conv2d(in_c, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.stages = nn.Sequential(
            *[nn.Sequential(
                ResMLPBlock(hidden_dim, self.token_dim, expansion_factor)
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
        out = einops.rearrange(out, 'b c w h -> b (w h) c')
        out = self.stages(out)
        out = einops.rearrange(out, 'b p c -> b c p')      # p: patch numbers
        out = einops.reduce(out, 'b c p -> b c', 'mean')
        out = self.head(out)
        return out


if __name__ == '__main__':
    image_size = 224
    patch_size = 8
    x = torch.rand(8, 3, image_size, image_size)
    model = ResMLP((16, 16), (224, 224), 3, 224, 4, 1, 5)
    # model = ResMLP((8, 8), (32, 32), 3, 32, 4, 2, 10)
    print(model)

    output = model(x)
    print(output.shape)