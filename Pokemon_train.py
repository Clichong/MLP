import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from Pokemon import Pokemon
from mlp_models import *
import random

epoch_size = 50
learning_rate = 1e-3
batch_size = 32
resize = 224
root = 'pokemon'
mdl_file = './checkpoint/pokemon_vip.mdl'
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

train_data = Pokemon(root=root, resize=resize, mode='train')
val_data = Pokemon(root=root, resize=resize, mode='val')
test_data = Pokemon(root=root, resize=resize, mode='test')

train_loader = DataLoader(train_data, batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
# model = ResMLP((16, 16), (224, 224), 3, 224, 4, 6, 5)
# model = RepMLPNet((16, 16), (224, 224), 3, 224, 4, 6, 5)
# model = aMLP((16, 16), (224, 224), 3, 224, 4, 6, 5)
model = SpinMLP((8, 8), (224, 224), 3, 224, 4, 6, 5, False)


model = model.to(device)
print(model)
print(mdl_file)

best_acc = 0
best_epoch = 0
infostat = {}
resume_flag = False
if resume_flag:
    # load checkpoint
    print("laod model...")
    checkpoint = torch.load(mdl_file)
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    best_epoch = checkpoint['epoch']
    print("best_acc: ", checkpoint['acc'], "best_epoch: ", checkpoint['epoch'])
# model.load_state_dict(torch.load(mdl_file)['net'])
# print(model)

crition = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epoch_size):

    # 训练集训练
    model.train()
    for batchidx, (image, label) in enumerate(train_loader):

        image = image.to(device)
        label = label.to(device)

        logits = model(image)
        loss = crition(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batchidx%5 == 0:
            print("epoch:{}/{}, batch:{}/{}, loss:{}"
                  .format(epoch+1, epoch_size, batchidx, len(train_loader), loss))

    # 测试集挑选
    model.eval()
    correct = 0
    for image, label in val_loader:

        image = image.to(device)
        label = label.to(device)

        with torch.no_grad():
            logits = model(image)
            pred = logits.argmax(dim=1)

        correct += torch.eq(pred, label).sum().float().item()

    acc = correct/len(val_data)
    infostat[epoch] = acc
    print("epoch:{}, acc:{}".format(epoch+1, acc))

    if acc >= best_acc:
        best_acc = acc
        best_epoch = epoch
        checkpoint = {
            'net': model.state_dict(),
            'acc': best_acc,
            'epoch': best_epoch,
        }
        torch.save(checkpoint, mdl_file)
        print("[get best epoch]- best_acc:{}, best_epoch:{}".format(best_acc, best_epoch+1))

    print(infostat)
    print("best_acc: ", checkpoint['acc'], "best_epoch: ", checkpoint['epoch'])

