import torch
from torchvision.models import resnet18
from thop import profile
from mlp_models import *
from mlp_net import *
from torchvision.models import resnet50

# model = MlpMixer((16, 16), (224, 224), 3, 224, 224*2, 224*4, 6, 5)
# model = S2MLPv1((16, 16), (224, 224), 3, 224, 4, 6, 5)
# model = S2MLPv2((16, 16), (224, 224), 3, 224, 4, 6, 5)
# model = ViP((16, 16), (224, 224), 3, 224, 16, 4, 6, 5)
# model = ASMLP((16, 16), (224, 224), 3, 224, 4, 6, 5)
# model = CycleMLP((16, 16),(224, 224), 3, 224, 4, 6, 5)
# model = HireMLP((16, 16), (224, 224), 3, 224, 4, 6, 5)
# model = SparseMLP((16, 16), (224, 224), 3, 224, 4, 6, 5)
# model = ConvMLP((16, 16), (224, 224), 3, 224, 4, 6, 5)
# model = gMLP((16, 16), (224, 224), 3, 224, 4, 6, 5)
# model = aMLP((16, 16), (224, 224), 3, 224, 4, 6, 5)
# model = ResMLP((16, 16), (224, 224), 3, 224, 4, 6, 5)
# model = RepMLPNet((16, 16), (224, 224), 3, 224, 4, 6, 5)
# model = SpinMLP((16, 16), (224, 224), 3, 224, 4, 6, 5, False)

# model = ASMLPNet()
# model = S2MLPv1Net()
# model = S2MLPv2Net()
# model = CycleMLPNet()
# model = SpinMLPNet()
model = StageMLP()
# model = resnet50()

input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, inputs=(input, ))

print('flops:{} G'.format(flops / 1000000000))
print('params:{} M'.format(params / 1000000))