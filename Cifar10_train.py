'''Train CIFAR10 with PyTorch.'''
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import random
import os
import argparse
import logging

from cnn_models import *
from mlp_models import *
from utils import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default='spinmlp', type=str, help='choose the model')
parser.add_argument('--patch_size', default=8, type=int, help='the small photo patch size')
parser.add_argument('--resize', default=32, type=int, help='train photo size')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='number of start epochs to run')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize')
parser.add_argument('--root', default='./cifar/cifar10', type=str, metavar='PATH', help='path to dataset')
parser.add_argument('--device', default='1', type=str, metavar='PATH', help='path to dataset')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')   # add --resume to perform
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_acc = 0    # best test accuracy
best_epoch = 0  # best test epoch
acc_lists = []
mdl_file = './record/checkpoint/cifar_{}.mdl'.format(args.model)
rec_file = './record/log/cifar_{}_log.txt'.format(args.model)
pho_file = './record/chart/cifar_{}_log.png'.format(args.model)

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


# add logger record
ALL_LOG_FORMAT = "%(message)s"

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

tx_handler = logging.FileHandler(filename=rec_file, mode='w+')
tx_handler.setLevel(logging.INFO)
tx_handler.setFormatter(logging.Formatter(ALL_LOG_FORMAT))

logger.addHandler(tx_handler)


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root=args.root, train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root=args.root, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# choose the model
if args.model == 'mlpmixer':
    net = MlpMixer((8, 8), (32, 32), 3, 32, 32*2, 32*4, 2, 10)
elif args.model == 's2mlpv1':
    net = S2MLPv1((8, 8), (32, 32), 3, 32, 4, 2, 10)
elif args.model == 's2mlpv2':
    net = S2MLPv2((8, 8), (32, 32), 3, 32, 4, 2, 10)
elif args.model == 'vip':
    net = ViP((8, 8), (32, 32), 3, 32, 8, 4, 2, 10)
elif args.model == 'asmlp':
    net = ASMLP((8, 8), (32, 32), 3, 32, 4, 2, 10)
elif args.model == 'cyclemlp':
    net = CycleMLP((8, 8), (32, 32), 3, 32, 4, 2, 10)
elif args.model == 'hiremlp':
    net = HireMLP((8, 8), (32, 32), 3, 32, 4, 2, 10)
elif args.model == 'sparsemlp':
    net = SparseMLP((8, 8), (32, 32), 3, 32, 4, 2, 10)
elif args.model == 'convmlp':
    net = ConvMLP((8, 8), (32, 32), 3, 32, 4, 2, 10)
elif args.model == 'gmlp':
    net = gMLP((8, 8), (32, 32), 3, 32, 4, 2, 10)
elif args.model == 'amlp':
    net = aMLP((8, 8), (32, 32), 3, 32, 4, 2, 10)
elif args.model == 'resmlp':
    net = ResMLP((8, 8), (32, 32), 3, 32, 4, 2, 10)
elif args.model == 'repmlpnet':
    net = RepMLPNet((8, 8), (32, 32), 3, 32, 4, 2, 10)
elif args.model == 'spinmlp':
    net = SpinMLP((8, 8), (32, 32), 3, 32, 4, 2, 10, False)
else:
    logger.error('No model name %s', args.model)

# Model
# net = MlpMixer((8, 8), (32, 32), 3, 32, 32*2, 32*4, 2, 10)
# net = S2MLPv1((8, 8), (32, 32), 3, 32, 4, 2, 10)
# net = S2MLPv2((8, 8), (32, 32), 3, 32, 4, 2, 10)
# net = ViP((8, 8), (32, 32), 3, 32, 8, 4, 2, 10)
# net = ASMLP((8, 8), (32, 32), 3, 32, 4, 2, 10)
# net = CycleMLP((8, 8), (32, 32), 3, 32, 4, 2, 10)
# net = HireMLP((8, 8), (32, 32), 3, 32, 4, 2, 10)
# net = SparseMLP((8, 8), (32, 32), 3, 32, 4, 2, 10)
# net = ConvMLP((8, 8), (32, 32), 3, 32, 4, 2, 10)
# net = gMLP((8, 8), (32, 32), 3, 32, 4, 2, 10)
# net = ResMLP((8, 8), (32, 32), 3, 32, 4, 2, 10)
# net = RepMLPNet((8, 8), (32, 32), 3, 32, 4, 2, 10)
# net = aMLP((8, 8), (32, 32), 3, 32, 4, 2, 10)
# net = SpinMLP((8, 8), (32, 32), 3, 32, 4, 2, 10)

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
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
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
        if batch_idx % 100 == 0:
            train_info = "epoch:{}/{}, batch:{}/{}, loss:{}, acc:{} ({}/{})"\
                  .format(epoch + 1, args.epochs, batch_idx, len(trainloader), train_loss / (batch_idx + 1),
                          correct / total, correct, total)
            print(train_info)
            logger.info(train_info)


def test(epoch):
    global best_acc
    global best_epoch
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    test_info = "Test ==> epoch:{}/{}, loss:{}, acc:{} ({}/{})"\
          .format(epoch + 1, args.epochs, test_loss/len(testloader),
                  correct/total, correct, total)
    print(test_info)
    logger.info(test_info)

    # Save checkpoint.
    acc = 100.*correct/total
    acc_lists.append(acc)
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
    # print(infostat)


for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
    train(epoch)
    test(epoch)
    scheduler.step()
    print("best_acc: ", best_acc, "best_epoch: ", best_epoch)

# draw photo
list_to_draw(acc_lists, filename=pho_file, title='{}_acc'.format(args.model), xlabel='Epochs', ylabel='Accuracy')
