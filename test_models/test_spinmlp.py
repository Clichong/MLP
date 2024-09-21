import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch import nn, optim
from torch.utils.data import DataLoader
from Pokemon import Pokemon
import numpy as np
import torchvision
import torchvision.transforms as transforms
import random
import os
import argparse

from cnn_models import *
from mlp_models import *

parser = argparse.ArgumentParser(description='PyTorch Pokemon Training')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='number of start epochs to run')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--root', default='../pokemon', type=str, metavar='PATH', help='path to dataset')
parser.add_argument('--device', default='1', type=str, metavar='PATH', help='path to dataset')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')   # add --resume to perform
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0    # best test accuracy
best_epoch = 0  # best test epoch
infostat = {}
mdl_file = './checkpoint/pokemon_spinmlp.mdl'
batch_size = 32
resize = 224
SEED = 42

# 应用不同的种子产生可复现的结果
def seed_everything(SEED=42):
    random.seed(SEED)
    # np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True  # keep True if all the input have same size.
seed_everything(SEED=SEED)

train_data = Pokemon(root=args.root, resize=resize, mode='train')
val_data = Pokemon(root=args.root, resize=resize, mode='val')
test_data = Pokemon(root=args.root, resize=resize, mode='test')

train_loader = DataLoader(train_data, batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net = MlpMixer((16, 16), (224, 224), 3, 224, 224*2, 224*4, 6, 5)
# net = S2MLPv1((16, 16), (224, 224), 3, 224, 4, 6, 5)
# net = S2MLPv2((16, 16), (224, 224), 3, 224, 4, 6, 5)
# net = ViP((16, 16), (224, 224), 3, 224, 16, 4, 6, 5)
# net = ASMLP((16, 16), (224, 224), 3, 224, 4, 6, 5)
# net = CycleMLP((16, 16),(224, 224), 3, 224, 4, 6, 5)
# net = HireMLP((16, 16), (224, 224), 3, 224, 4, 6, 5)
# net = SparseMLP((16, 16), (224, 224), 3, 224, 4, 6, 5)
# net = ConvMLP((16, 16), (224, 224), 3, 224, 4, 6, 5)
# net = gMLP((16, 16), (224, 224), 3, 224, 4, 6, 5)
# net = ResMLP((16, 16), (224, 224), 3, 224, 4, 6, 5)
# net = RepMLPNet((16, 16), (224, 224), 3, 224, 4, 6, 5)
# net = aMLP((16, 16), (224, 224), 3, 224, 4, 6, 5)
net = SpinMLP((16, 16), (224, 224), 3, 224, 4, 6, 5, False)

net = net.to(device)
print(net)
print(mdl_file)

# perform mutil gpu compute
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# reload trained model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(mdl_file)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print("best_acc: ", checkpoint['acc'], "best_epoch: ", checkpoint['epoch'])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if batch_idx % 5 == 0:
            print("epoch:{}/{}, batch:{}/{}, loss:{}, acc:{} ({}/{})"
                  .format(epoch + 1, args.epochs, batch_idx, len(train_loader), train_loss / (batch_idx + 1),
                          correct / total, correct, total))


def test(epoch):
    global best_acc
    global best_epoch
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    print("Test ==> epoch:{}/{}, loss:{}, acc:{} ({}/{})"
          .format(epoch + 1, args.epochs, test_loss/len(test_loader),
                  correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    infostat[epoch] = acc
    if acc >= best_acc:
        print('Saving..')
        checkpoint = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, mdl_file)
        best_acc = acc
        best_epoch = epoch
        # print("best_acc: ", checkpoint['acc'], "best_epoch: ", checkpoint['epoch'])
    print(infostat)


for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
    train(epoch)
    test(epoch)
    # scheduler.step()
    print("best_acc: ", best_acc, "best_epoch: ", best_epoch)

