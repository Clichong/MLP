import torch
import torch.nn as nn
import einops
from timm.models.layers import trunc_normal_, to_2tuple

class MlpBlock(nn.Module):

    def __init__(self, hidden_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class MixerBlock(nn.Module):

    def __init__(self, token_dim, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.token_mixing = MlpBlock(token_dim, tokens_mlp_dim)
        self.channel_mixing = MlpBlock(hidden_dim, channels_mlp_dim)

    def forward(self, x):
        y = self.layer_norm(x)
        y = y.permute(0, 2, 1)  # hidden_dim, token_dim: 512, 49
        y = self.token_mixing(y)
        y = y.permute(0, 2, 1)
        x = x + y
        y = self.layer_norm(y)  # token_dim, hidden_dim: 49, 512
        y = self.channel_mixing(y)
        return y + x


class MlpMixer(nn.Module):

    def __init__(self, patch_size, image_size, in_c, hidden_dim, tokens_mlp_dim, channels_mlp_dim, num_blocks, num_classes):
        super().__init__()
        self.num_blocks = num_blocks
        self.token_dim = (image_size[0]//patch_size[0])*(image_size[1]//patch_size[1])
        self.patch_embed = nn.Conv2d(in_c, hidden_dim, kernel_size=patch_size, stride=patch_size)
        # self.mixer_block = MixerBlock(self.token_dim, hidden_dim, tokens_mlp_dim, channels_mlp_dim)
        self.stages = nn.Sequential(
            *[nn.Sequential(
                MixerBlock(self.token_dim, hidden_dim, tokens_mlp_dim, channels_mlp_dim)
            ) for i in range(self.num_blocks)]
        )
        self.proj = nn.Linear(hidden_dim, num_classes)
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
        out = einops.rearrange(out, 'b c h w -> b (h w) c')
        out = self.stages(out)
        out = out.mean(1)
        out = self.proj(out)
        return out


if __name__ == '__main__':
    image_size = 224
    x = torch.rand(8, 3, image_size, image_size)
    model = MlpMixer((32, 32), (image_size, image_size), 3, 224, 224*2, 224*4, 6, 5)
    # model = MlpMixer((8, 8), (image_size, image_size), 3, 32, 32, 32, 8, 10)
    print(model)

    output = model(x)
    print(output.shape)

