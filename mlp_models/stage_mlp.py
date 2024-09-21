import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, to_2tuple, DropPath


class MLP(nn.Module):

    def __init__(self, hidden_dim, expansion_factor, drop=0.):
        super().__init__()
        self.fc1 = nn.Conv2d(hidden_dim, hidden_dim * expansion_factor, kernel_size=(1, 1))
        self.gelu = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim * expansion_factor, hidden_dim, kernel_size=(1, 1))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)
        return out


class DynaMixerOperation(nn.Module):

    def __init__(self, N, D, d=10):
        super().__init__()

        self.N = N
        self.D = D
        self.d = d

        self.fc_ND = nn.Linear(D, d)
        self.fc_Nd = nn.Linear(N*d, N*N)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input):

        B, D, N = input.shape

        # Dynamic mixing matrix generation
        input = rearrange(input, 'b d n -> b n d')
        p = self.fc_ND(input)
        p = p.reshape(-1, 1, N*self.d)
        p = self.fc_Nd(p)
        p = p.reshape(-1, N, N)
        p = self.softmax(p)

        out = torch.matmul(p, input)
        out = rearrange(out, 'b n d -> b d n')
        return out


class DynaMixerBlock(nn.Module):

    def __init__(self, channels, imagesize):
        super().__init__()

        h_size = imagesize[0]
        w_size = imagesize[1]
        self.dynamixer_op_h = DynaMixerOperation(w_size, channels)
        self.dynamixer_op_w = DynaMixerOperation(h_size, channels)
        self.proj_c = nn.Conv2d(channels, channels, kernel_size=(1, 1))
        self.proj_o = nn.Conv2d(channels, channels, kernel_size=(1, 1))

    def forward(self, input):

        b, c, h, w = input.shape

        # row mixing
        # Y_h = torch.zeros([b, c, h, w])
        Y_h = input.clone()
        for i in range(h):
            Y_h[:, :, i, :] = self.dynamixer_op_h(input[:, :, i, :])     # (b c w)

        # column mixing
        # Y_w = torch.zeros([b, c, h, w])
        Y_w = input.clone()
        for i in range(w):
            Y_w[:, :, :, i] = self.dynamixer_op_w(input[:, :, :, i])     # (b c h)

        # channel mixing
        Y_c = self.proj_c(input)

        Y_out = Y_h + Y_w + Y_c
        out = self.proj_o(Y_out)

        return out


class LayerNorm(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.ln = nn.Sequential(
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(hidden_dim),
            Rearrange('b h w c -> b c h w'),
        )

    def forward(self, x):
        # x.size: (b c h w)
        out = self.ln(x)
        return out


class StageMLPBlock(nn.Module):

    def __init__(self, hidden_dim, image_size, expansion_factor, drop, drop_path_rate):
        super().__init__()
        self.model_p1 = nn.Sequential(
            LayerNorm(hidden_dim),
            DynaMixerBlock(hidden_dim, to_2tuple(image_size)),
        )
        self.model_p2 = nn.Sequential(
            LayerNorm(hidden_dim),
            MLP(hidden_dim, expansion_factor, drop)
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        out = self.model_p1(x)
        x = x + self.drop_path(out)
        out = self.model_p2(x)
        x = x + self.drop_path(out)
        return x


class DownSample(nn.Module):

    def __init__(self, in_c, out_c, patch_size):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=to_2tuple(patch_size), stride=(2, 2), padding=1)

    def forward(self, x):
        out = self.conv(x)
        return out


class StageMLP(nn.Module):

    def __init__(self,
                 patch_size=[4, 3, 3, 3, 3],    # ignore the last numbers
                 image_size=(56, 28, 14, 7),
                 in_c=3,
                 hidden_dim=[96, 192, 384, 768, 1440],
                 expansion_factor=[2, 2, 2, 2],
                 num_blocks=[1, 1, 3, 1],
                 drop=0.6,
                 drop_path_rate=0.2,
                 num_classes=101):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_stages = len(num_blocks)
        self.image_size = image_size
        self.patch_embed = nn.Conv2d(in_c, hidden_dim[0],
                                     kernel_size=to_2tuple(patch_size[0]), stride=to_2tuple(patch_size[0]))

        # learn from ConvNeXt
        down_stage = []
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.num_blocks))]
        cur = 0
        for i in range(self.num_stages):
            down_stage.append(DownSample(hidden_dim[i], hidden_dim[i + 1], patch_size[i + 1]))
            mlp_stage = nn.Sequential(
                *[StageMLPBlock(hidden_dim[i], image_size[i], expansion_factor[i],
                                drop=drop, drop_path_rate=dp_rates[cur + j])
                  for j in range(self.num_blocks[i])]
            )
            self.__setattr__('stage{}'.format(i), mlp_stage)
            cur += self.num_blocks[i]

        self.down_stage = nn.ModuleList(down_stage)
        self.head = nn.Linear(hidden_dim[-1], num_classes)
        self._init_weights()

    def forward(self, x):
        out = self.patch_embed(x)
        outputs = []    # get four feature maps
        for i in range(self.num_stages):
            mlp_stage = self.__getattr__('stage{}'.format(i))
            down_stage = self.down_stage[i]
            out = mlp_stage(out)        # get stage feature
            outputs.append(out)
            out = down_stage(out)       # down sample operate

        # feaure maps fusion
        for i in range(self.num_stages):
            outputs[i] = F.interpolate(outputs[i], size=to_2tuple(self.image_size[-1]), mode='bilinear')
        output = torch.cat((outputs[0], outputs[1], outputs[2], outputs[3]), dim=1)
        output = einops.reduce(output, 'b c h w -> b c', 'mean')
        output = self.head(output)
        return output

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


if __name__ == '__main__':

    x = torch.rand([8, 3, 224, 224])
    x = torch.autograd.Variable(x, requires_grad=True)

    model = StageMLP()
    print(model)

    output = model(x)
    print(output.shape)

    # for out in output:
    #     print(out.shape)

    # out = output[0][0][0][0][0]
    # print(out)

    # out.backward()
    # print(f"input grad: \n{x.grad}\n")















