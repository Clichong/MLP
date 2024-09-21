import torch
from utils import *
import argparse
import torchvision
import torch.nn as nn


net = torchvision.models.resnet50(pretrained=True)
net.fc = nn.Linear(2048, 101)
print(net)