import os
import numpy as np
import torch
import cv2
import torchvision
from torch.utils.data import Dataset


class ReadYOLO(Dataset):
    def __init__(self, dateset_address='./dataset', type="classification", phase="train", trans=None, device=None):
        '''
        :param type:输入需要读取的模型类型
            "classification":分类
            "objectdetection":目标检测
            facedetection”:人脸检测
        :param phase:数据类型
            "train":训练数据
            "valid":验证数据
            "test":测试数据
        :param trans:是否进行图像增强
        :param device:CPU/GPU
        '''
        super(ReadYOLO, self).__init__()
        self.device = device
        self.type = type
        self.trans = trans
        self.phase = phase
        self.dateset_address = dateset_address

        # 文件外调用时
        self.labels = os.listdir(os.path.join(self.dateset_address, self.phase, 'label'))  # 标签文件根目录所有.txt文件的名字
        self.imgs = os.listdir(os.path.join(self.dateset_address, self.phase, 'img'))  # 图片文件根目录所有.jpg文件的名字
        # # 此文件内测试使用
        # self.labels = os.listdir(os.path.join('../dataset', self.phase, 'label'))
        # self.imgs = os.listdir(os.path.join('../dataset', self.phase, 'img'))
        self.img_names = list(map(lambda x: x.split('.')[0], self.imgs))  # 图片文件根目录不带.jpg的所有文件的名字

    # 重写 len(readYOLO) 方法
    def __len__(self):
        return len(self.labels)

    # 重写 readYOLO[i] 方法
    def __getitem__(self, item):
        '''
        :param item:
        :return: tensor格式的image矩阵，label的类别坐标矩阵（目标检测）
        '''
        picture = None  # image矩阵
        list_target = []  # 把txt文件内的行变成列表储存，用于后续拼接成array

        # 已知标签的item，获取第item个图片（获取txt文件对应的jpg文件）（数据集很大的时候，若存在无标注图片，直接返回会对应错误）
        img = self.imgs[list(map(lambda x: x == self.labels[item].split('.')[0], self.img_names)).index(True)]
        img_dir = os.path.join(self.dateset_address, self.phase, 'img', img)  # Invoke
        # img_dir = os.path.join('../dataset', self.phase, 'img', img)  # test
        picture = cv2.imread(img_dir)  # array=[w,h,3]

        with open(os.path.join(self.dateset_address, self.phase, 'label', self.labels[item]), 'r') as f:
            # 方便目标检测等类型
            for line in f.readlines():
                # 判断此行是否为空
                if len(line.strip('\n')) > 0:
                    nums = line.strip().split(' ')
                    # map全部转换为浮点数 [*map()]解压后存入列表==list(map())
                    li = [*map(lambda x: float(x), nums)]
                    list_target.append(li)
        if len(list_target) == 0:
            array_target = np.array([])
        else:
            # 列表变为numpy.array数组格式，reshape确保格式正确
            array_target = np.concatenate(list_target, axis=0).reshape(len(list_target), -1)

        # 是否图像增强
        if self.trans:
            picture, array_target = self.trans(picture, array_target)  # picture变为[3,w,h]
            # print(picture.shape)
            # print(picture.unsqueeze(0).shape)
            # 返回结果 picture升一维变为[b,c,w,h]
            return picture.unsqueeze(0).to(self.device), torch.from_numpy(array_target).to(self.device)
        else:
            trans = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize(size=(224, 224)),  # 若批处理数>1，则需要图片大小一致
                torchvision.transforms.ToTensor()]  # (H,W,C)转换成形状为[C,H,W]
            )
            picture = trans(picture)
            # print(picture.shape)
            return picture.unsqueeze(0).to(self.device), torch.from_numpy(array_target).to(self.device)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from Augmentation.data_augment import DataAugment


    # 定义取数据方法
    def colle(batch):  # 重写选择数据集的方式
        # 假设一个batch_size=2:那么batch的shape就是((picture1_tensor, picture1_label), (picture2_tensor, picture2_label))
        imgs, targets = list(zip(*batch))  # 通过解压batch把多张图片的picture_tensor放在一起，picture_label放在一起
        imgs = torch.cat(imgs, dim=0)
        targets = torch.cat(targets, dim=0)
        return imgs, targets


    data_augment = DataAugment()
    dataset = ReadYOLO(trans=data_augment)  # 进行图像增强
    dataset_no = ReadYOLO()  # 无图像增强

    data = iter(DataLoader(dataset, batch_size=4, drop_last=False, collate_fn=colle))
    data_no = iter(DataLoader(dataset_no, batch_size=4, drop_last=False, collate_fn=colle))
    imgs, labels = next(data)
    print(imgs.shape)
    print(imgs)
    print(labels.shape)
    print(labels)

    imgs, labels = next(data_no)
    print(imgs.shape)
    print(imgs)
    print(labels.shape)
    print(labels)
