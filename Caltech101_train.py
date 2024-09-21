from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
import cv2
import os
import torch
import torchvision
import numpy as np
import torch.optim as optim
import random
from imutils import paths

from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse

from cnn_models import *
from mlp_models import *
from mlp_net import *
from utils import *

# set matplotlib style
# matplotlib.style.use('ggplot')

parser = argparse.ArgumentParser(description='PyTorch Caltech101 Training')
parser.add_argument('--model', default='stagemlp', type=str, help='choose the model')
parser.add_argument('--patch_size', default=16, type=int, help='the small photo patch size')
parser.add_argument('--resize', default=224, type=int, help='train photo size')
parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='number of start epochs to run')
parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize')
parser.add_argument('--root', default='./caltech/caltech101/101_ObjectCategories/', type=str, metavar='PATH', help='path to dataset')
parser.add_argument('--device', default='1,0', type=str, metavar='PATH', help='path to dataset')
parser.add_argument('--cls_nums', default=101, type=int, help='model to class numbers')
parser.add_argument('--blk_nums', default=6, type=int, help='model to block numbers')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

# 按照PCI_BUS_ID顺序从0开始排列GPU设备
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_acc = 0    # best test accuracy
best_epoch = 0  # best test epoch
acc_lists = []
trainloss_lists = []
testloss_lists = []
mdl_file = './record/checkpoint/caltech101/caltech101_{}.mdl'.format(args.model)
rec_file = './record/log/caltech101/caltech101_{}_log.txt'.format(args.model)
pho_file = './record/chart/caltech101/caltech101_{}_log.png'.format(args.model)
trainloss_imgfile = './record/loss/caltech101/caltech101_{}_trainloss.png'.format(args.model)
testloss_imgfile  = './record/loss/caltech101/caltech101_{}_testloss.png'.format(args.model)

# make seed
def seed_everything(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True  # keep True if all the input have same size.
SEED = 42
seed_everything(SEED=SEED)


# add logger record
ALL_LOG_FORMAT = "%(message)s"

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

tx_handler = logging.FileHandler(filename=rec_file, mode='w+')
tx_handler.setLevel(logging.INFO)
tx_handler.setFormatter(logging.Formatter(ALL_LOG_FORMAT))

logger.addHandler(tx_handler)

# pre label and data
image_paths = list(paths.list_images(args.root))
data = []
labels = []
for image_path in image_paths:
    label = image_path.split(os.path.sep)[-2]
    if label == 'BACKGROUND_Google':
        continue
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    data.append(image)
    labels.append(label)

data = np.array(data)
labels = np.array(labels)
gle = LabelEncoder()
genre_labels = gle.fit_transform(labels)
genre_mappings = {label: index for index, label in enumerate(gle.classes_)}
print(genre_mappings)

# divide the data into train, validation, and test set
(x_train, x_val, y_train, y_val) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=SEED)
print(f"x_train examples: {x_train.shape} x_val examples: {x_val.shape}")

# custom dataset
class ImageDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None, mappings=None):
        self.X = images
        self.y = labels
        self.transforms = transforms
        self.mappings = mappings

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        data = self.X[i][:]
        label = self.mappings[self.y[i]]

        if self.transforms:
            data = self.transforms(data)

        if self.y is not None:
            return data, label
        else:
            return data


# data loading
train_transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((args.resize, args.resize)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])
val_transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((args.resize, args.resize)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

train_data = ImageDataset(x_train, y_train, train_transform, genre_mappings)
val_data = ImageDataset(x_val, y_val, val_transform, genre_mappings)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)


# choose the model
num_classes = args.cls_nums
num_blocks  = args.blk_nums
if args.model == 'mlpmixer':
    net = MlpMixer((16, 16), (224, 224), 3, 224, 512, 2048, num_blocks, num_classes)
elif args.model == 's2mlpv1':
    net = S2MLPv1((16, 16), (224, 224), 3, 224, 4, num_blocks, num_classes)
elif args.model == 's2mlpv2':
    net = S2MLPv2((16, 16), (224, 224), 3, 224, 4, num_blocks, num_classes)
elif args.model == 'vip':
    net = ViP((16, 16), (224, 224), 3, 224, 16, 4, num_blocks, num_classes)
elif args.model == 'asmlp':
    net = ASMLP((16, 16), (224, 224), 3, 224, 4, num_blocks, num_classes)
elif args.model == 'cyclemlp':
    net = CycleMLP((16, 16),(224, 224), 3, 224, 4, num_blocks, num_classes)
elif args.model == 'hiremlp':
    net = HireMLP((16, 16), (224, 224), 3, 224, 4, num_blocks, num_classes)
elif args.model == 'sparsemlp':
    net = SparseMLP((16, 16), (224, 224), 3, 224, 4, num_blocks, num_classes)
elif args.model == 'convmlp':
    net = ConvMLP((16, 16), (224, 224), 3, 224, 4, num_blocks, num_classes)
elif args.model == 'gmlp':
    net = gMLP((16, 16), (224, 224), 3, 224, 4, num_blocks, num_classes)
elif args.model == 'amlp':
    net = aMLP((16, 16), (224, 224), 3, 224, 4, num_blocks, num_classes)
elif args.model == 'resmlp':
    net = ResMLP((16, 16), (224, 224), 3, 224, 4, num_blocks, num_classes)
elif args.model == 'repmlpnet':
    net = RepMLPNet((16, 16), (224, 224), 3, 224, 4, num_blocks, num_classes)
elif args.model == 'spinmlp':
    net = SpinMLP((16, 16), (224, 224), 3, 224, 4, num_blocks, num_classes, False)
elif args.model == 'asmlpnet':
    net = ASMLPNet()
elif args.model == 's2mlpv1net':
    net = S2MLPv1Net()
elif args.model == 's2mlpv2net':
    net = S2MLPv2Net()
elif args.model == 'cyclemlpnet':
    net = CycleMLPNet()
elif args.model == 'spinmlpnet':
    net = SpinMLPNet()
elif args.model == 'stagemlp':
    net = StageMLP(drop=0.6, drop_path_rate=0.2, num_classes=101)
elif args.model == 'resnet50':
    net = torchvision.models.resnet50(pretrained=True)
    net.fc = nn.Linear(2048, 101)
else:
    logger.error('No model name %s', args.model)



net = net.to(device)
print(net)
print(mdl_file)

# perform mutil gpu compute
if torch.cuda.is_available():
    print("\ntorch.cuda.is_available is Ture.\n")
    # assert torch.cuda.is_available(), "torch.cuda.is_available is False"
    if torch.cuda.device_count() > 1:
        print("Use device: ", torch.cuda.device_count(), "GPU \n")
    # 默认使用所有卡
    net = torch.nn.DataParallel(net)
    # cudnn.benchmark = True


# reload trained model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(mdl_file), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(mdl_file)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print("best_acc: ", checkpoint['acc'], "best_epoch: ", checkpoint['epoch'])


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

# 启用异常检测以查找未能计算其梯度的操作
torch.autograd.set_detect_anomaly(True)

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
        if batch_idx % 10 == 0:
            train_info = "epoch:{}/{}, batch:{}/{}, loss:{}, acc:{} ({}/{})"\
                  .format(epoch + 1, args.epochs, batch_idx, len(train_loader), train_loss / (batch_idx + 1),
                          correct / total, correct, total)
            print(train_info)
            logger.info(train_info)

    # add train loss
    trainloss_lists.append(train_loss / len(train_loader))


def test(epoch):
    global best_acc
    global best_epoch
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
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
          .format(epoch + 1, args.epochs, test_loss/len(val_loader),
                  correct/total, correct, total)
    print(test_info)
    logger.info(test_info)

    # Save checkpoint.
    testloss_lists.append(test_loss / len(val_loader))
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
    # print(infostat)


for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
    train(epoch)
    test(epoch)
    logger.info("best_acc: {} best_epoch: {}".format(best_acc, best_epoch))

# draw photo
list_to_draw(acc_lists, filename=pho_file, title='{}_acc'.format(args.model),
             xlabel='Epochs', ylabel='Accuracy')
list_to_draw(trainloss_lists, filename=trainloss_imgfile, title='{}_trainloss'.format(args.model),
             xlabel='Epochs', ylabel='Train Loss')
list_to_draw(testloss_lists, filename=testloss_imgfile, title='{}_testloss'.format(args.model),
             xlabel='Epochs', ylabel='Test Loss')

