import torch.backends.cudnn as cudnn

import torch
from torch import nn, optim

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from Pokemon import Pokemon
import random
import os
import argparse
import logging

from cnn_models import *
from mlp_models import *
from utils import *

parser = argparse.ArgumentParser(description='PyTorch Pokemon Training')
parser.add_argument('--model', default='spinmlp', type=str, help='choose the model')
parser.add_argument('--patch_size', default=16, type=int, help='the small photo patch size')
parser.add_argument('--resize', default=224, type=int, help='train photo size')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='number of start epochs to run')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize')
parser.add_argument('--root', default='pokemon', type=str, metavar='PATH', help='path to dataset')
parser.add_argument('--device', default='0, 1', type=str, metavar='PATH', help='path to dataset')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')   # add --resume to perform
args = parser.parse_args()


best_acc = 0    # best test accuracy
best_epoch = 0  # best test epoch
acc_lists = []
mdl_file = './record/checkpoint/pokemon_{}.mdl'.format(args.model)
rec_file = './record/log/pokemon_{}_log.txt'.format(args.model)
pho_file = './record/chart/pokemon_{}_log.png'.format(args.model)

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


# choose the model
if args.model == 'mlpmixer':
    net = MlpMixer((16, 16), (224, 224), 3, 224, 512, 2048, 6, 5)
elif args.model == 's2mlpv1':
    net = S2MLPv1((16, 16), (224, 224), 3, 224, 4, 6, 5)
elif args.model == 's2mlpv2':
    net = S2MLPv2((16, 16), (224, 224), 3, 224, 4, 6, 5)
elif args.model == 'vip':
    net = ViP((16, 16), (224, 224), 3, 224, 16, 4, 6, 5)
elif args.model == 'asmlp':
    net = ASMLP((16, 16), (224, 224), 3, 224, 4, 6, 5)
elif args.model == 'cyclemlp':
    net = CycleMLP((16, 16),(224, 224), 3, 224, 4, 6, 5)
elif args.model == 'hiremlp':
    net = HireMLP((16, 16), (224, 224), 3, 224, 4, 6, 5)
elif args.model == 'sparsemlp':
    net = SparseMLP((16, 16), (224, 224), 3, 224, 4, 6, 5)
elif args.model == 'convmlp':
    net = ConvMLP((16, 16), (224, 224), 3, 224, 4, 6, 5)
elif args.model == 'gmlp':
    net = gMLP((16, 16), (224, 224), 3, 224, 4, 6, 5)
elif args.model == 'amlp':
    net = aMLP((16, 16), (224, 224), 3, 224, 4, 6, 5)
elif args.model == 'resmlp':
    net = ResMLP((16, 16), (224, 224), 3, 224, 4, 6, 5)
elif args.model == 'repmlpnet':
    net = RepMLPNet((16, 16), (224, 224), 3, 224, 4, 6, 5)
elif args.model == 'spinmlp':
    net = SpinMLP((16, 16), (224, 224), 3, 224, 4, 6, 5, False)
elif args.model == 'stagemlp':
    net = StageMLP()
else:
    logger.error('No model name %s', args.model)

# 1. use DataParallel
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# perform mutil gpu compute
if device.type == 'cuda':
    print("Cuda is available and use DataParallel")
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    # cudnn.benchmark = True
print(net)
print(mdl_file)

# 2, use DistributedDataParallel
# torch.distributed.init_process_group(backend="nccl")
#
# # 配置每个进程的gpu
# local_rank = torch.distributed.get_rank()
# torch.cuda.set_device(local_rank)
# device = torch.device("cuda", local_rank)

# 封装之前要把模型移到对应的gpu
# model = StageMLP()
# net = net.to(device)
# net = DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank)

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

# load data
train_data = Pokemon(root=args.root, resize=args.resize, mode='train')
val_data = Pokemon(root=args.root, resize=args.resize, mode='val')
test_data = Pokemon(root=args.root, resize=args.resize, mode='test')

train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
val_loader = DataLoader(val_data, args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, args.batch_size, shuffle=True)
# train_loader = DataLoader(train_data, args.batch_size,
#                           pin_memory=True, drop_last=True, sampler=DistributedSampler(train_data))
# val_loader = DataLoader(train_data, args.batch_size,
#                           pin_memory=True, drop_last=True, sampler=DistributedSampler(val_data))
# test_loader = DataLoader(train_data, args.batch_size,
#                           pin_memory=True, drop_last=True, sampler=DistributedSampler(test_data))



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
            train_info = "epoch:{}/{}, batch:{}/{}, loss:{}, acc:{} ({}/{})"\
                .format(epoch + 1, args.epochs, batch_idx, len(train_loader), train_loss / (batch_idx + 1),
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
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_info = "Test ==> epoch:{}/{}, loss:{}, acc:{} ({}/{})"\
        .format(epoch + 1, args.epochs, test_loss/len(test_loader), correct/total, correct, total)
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
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        torch.save(checkpoint, mdl_file)
        best_acc = acc
        best_epoch = epoch
        # print("best_acc: ", checkpoint['acc'], "best_epoch: ", checkpoint['epoch'])
    # print(acc_lists)


for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
    train(epoch)
    test(epoch)
    logger.info("best_acc: {} best_epoch: {}".format(best_acc, best_epoch))

# draw photo
list_to_draw(acc_lists, filename=pho_file, title='{}_acc'.format(args.model), xlabel='Epochs', ylabel='Accuracy')
