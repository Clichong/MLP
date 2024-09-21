import torch
import torchvision
from torch.utils.data import Dataset, DataLoader   # 注意是Dataset而不是dataset
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.utils import save_image
# from visdom import Visdom
import os, glob, random, csv, time


class Pokemon(Dataset):

    def __init__(self, root, resize, mode):

        super(Pokemon, self).__init__()

        self.root = root
        self.resize = resize

        self.image = []
        self.label = []

        # 创建一个字典存储类别与标签
        self.name2label = {}

        for name in os.listdir(root):
            # 判断文件名是否为目录
            if not os.path.isdir(os.path.join(root, name)):
                continue
            # 关键字的取值为当前的关键字个数
            self.name2label[name] = len(self.name2label.keys())
            # print(self.name2label.keys())
            # dict_keys(['bulbasaur', 'charmander', 'mewtwo', 'pikachu'])

        # print(self.name2label)
        # {'bulbasaur': 0, 'charmander': 1, 'mewtwo': 2, 'pikachu': 3, 'squirtle': 4}

        # 导入图像数据
        # self.load_csv('images.csv')
        self.image, self.label = self.load_csv('images.csv')

        # 设置train-val-test比例
        # nums: 700
        if mode == 'train':
            self.image = self.image[:int(0.6 * len(self.image))]
            self.label = self.label[:int(0.6 * len(self.label))]
        # nums: 233
        elif mode == 'val':
            self.image = self.image[int(0.6 * len(self.image)):int(0.8 * len(self.image))]
            self.label = self.label[int(0.6 * len(self.label)):int(0.8 * len(self.label))]
        # nums: 234
        elif mode == 'test':
            self.image = self.image[int(0.8 * len(self.image)):]
            self.label = self.label[int(0.8 * len(self.label)):]
        else:
            print("Error! 'Mode' has no such mode choice!")


    def __len__(self):

        return len(self.image)

    def __getitem__(self, item):

        # item = self.__len__()
        # print(" item: ", item)

        image = self.image[item]
        label = self.label[item]
        # print("image: ",image,"label: ",label)

        # 对图像进行预处理
        transform = transforms.Compose([
            # 转换为RGB图像
            lambda x: Image.open(x).convert('RGB'),
            # 重新确定尺寸
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            # 随机旋转角度
            transforms.RandomRotation(15),
            # 中心裁剪
            transforms.CenterCrop(self.resize),
            # 转换为Tensor格式
            transforms.ToTensor(),
            # 使数据分布在0附近
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        image = transform(image)
        # label是int形式，转换为tensor格式
        label = torch.tensor(label)

        return image, label


    # 导入csv样本数据
    def load_csv(self, csv_file):

        # 当没有csv数据文件时创建文件, 将数据集信息保存在一个csv_file文件中
        if not os.path.exists(os.path.join(self.root, csv_file)):

            # 用来存储图像路径信息
            image = []

            # 现查找数据集文件中的png，jpg，jpeg格式的全部图像，路径全部保存在image中
            for name in self.name2label.keys():
                # glob 模块用于查找符合特定规则的文件路径名
                image += glob.glob(os.path.join(self.root, name, '*.png'))
                image += glob.glob(os.path.join(self.root, name, '*.jpg'))
                image += glob.glob(os.path.join(self.root, name, '*.jpeg'))

            # 'E:\\学习\\机器学习\\数据集\\pokemon\\bulbasaur\\00000000.png'
            # print(image, len(image))

            # 随机打乱图像
            random.shuffle(image)

            # 截取绝对路径下的图像名字
            # name = next(iter(image))
            # name = name.split('\\')[-2]
            # print(name)   # charmander


            # 读写打开文件, 注意newline=''是为了不让存储的时候回车两行
            with open(csv_file, mode='w', newline='') as f:

                # 创建 csv 对象
                writer = csv.writer(f)

                for img in image:

                    # split: 对路径进行分割，以列表形式返回
                    # os.sep: 当前操作系统所使用的路径分隔符 windows->'\' linux 和 unix->'/'
                    # ['E:/学习/机器学习/数据集/pokemon', 'pikachu', '00000179.jpg']
                    # [-2]既提取了文件夹名字: 'pikachu'
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]

                    # 写入一行或多行数据
                    # 形式: E:\学习\机器学习\数据集\pokemon\charmander\00000185.png,1
                    writer.writerow([img, label])

                # print('writen into csv file:', csv_file)

        # 打开csv文件读取信息
        with open(csv_file) as f:

            # 创建两个list存储图像名字与标签
            image = []
            label = []

            # #创建 csv 对象,它是一个包含所有数据的列表，每一行为名字与标签,eg: charmander,1
            reader = csv.reader(f)

            # 循环赋值各行内容
            for row in reader:

                # 导入数据， 若没有设置newline=''会报错，因为回车了两行
                image.append(row[0])
                label.append(int(row[1]))

            # print(len(image), len(label))

        if len(image) == len(label):
            return image, label
        else:
            print("Error! len(image) != len(label) !")


    def denormalize(self, x_hat):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x_hat = (x-mean)/std
        # x = x_hat*std = mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean

        return x



def plot_image(img):

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img[i][0]*0.3081+0.1307, cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == '__main__':

    print("Pokemon dataset")
    root = 'E:\学习\机器学习\数据集\pokemon'
    viz = Visdom()

    train_data = Pokemon(root=root, resize=64, mode='train')
    # print(train_data.__len__())

    # image, label = next(iter(train_data))
    # print(image.shape, label)

    # 利用DataLoader加载数据集
    data = DataLoader(train_data, batch_size=64, shuffle=True)

    # 测试
    for epochodx, (image, label) in enumerate(data):
        # plot_image(train_data.denormalize(image))
        # time.sleep(5)

        save_image(image, os.path.join('sample', 'image-{}.png'.format(epochodx + 1)), nrow=8, normalize=True)
        # viz.images(image, nrow=8, win='batch', opts=dict(title='batch'))
        viz.images(train_data.denormalize(image), nrow=8, win='batch', opts=dict(title='batch'))
        time.sleep(5)