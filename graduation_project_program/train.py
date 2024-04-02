import os
from math import ceil
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torchvision
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from utils.myloss import MyLoss
from dataset.read_yolo_dataset import ReadYOLO
from Augmentation.data_augment import DataAugment

import logging
import argparse
import time

# 命令行执行传参
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--model', default="vgg11", help='model')  # 选择模型
# (alexnet; vgg11/13/16/19(_bn); googlenet; resnet18/34/50; densenet121/161; convnext_tiny/small)
# 所有地址相关变量放在一个文件中，方便上云管理
parser.add_argument('--dateset_address', default="./dataset", help='dateset_address')  # 数据集地址
parser.add_argument('--weights_address', default="./weights", help='weights_address')  # 模型参数存储地址
parser.add_argument('--log_address', default="./log", help='log_address')  # 日志存储地址
parser.add_argument('--visualization_address', default="./visualization", help='visualization_address')  # 可视化地址
# 训练模型相关参数设置
parser.add_argument('--num_classes', default=2, type=int, help='num_classes')  # 目标分类类别数
parser.add_argument('--train_rate', default=0.8, type=float, help='train_rate')  # 训练集切分比例
parser.add_argument('--lr', default=0.001, type=float, help='learning rate of model')  # 学习率
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')  # 动量
parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
parser.add_argument('--epochs', default=20, type=int, help='epochs')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')  # SGD的权重衰减
args = parser.parse_args()

# 如果存储文件夹不存在，则创建
if not os.path.exists(args.weights_address):
    os.makedirs(args.weights_address)
if not os.path.exists(args.log_address):
    os.makedirs(args.log_address)
if not os.path.exists(args.visualization_address):
    os.makedirs(args.visualization_address)

# 创建全局device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取数据集
size = (224, 224)
data_augment = DataAugment(size=size)  # 数据增强实例化
dataset = ReadYOLO(dateset_address=args.dateset_address, phase='train', trans=data_augment, device=device)  # 读取数据集实例化
picture_num = len(dataset)  # 获取图片总数

# 模型实例化
kwargs = {"num_classes": args.num_classes}
net = torchvision.models.vgg16()
creat_model = f"net = torchvision.models.{args.model}(**{kwargs})"
exec(creat_model)
net = net.to(device=device)

# 迭代器和损失函数优化器实例化
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# loss = MyLoss()  # 等价于loss = nn.CrossEntropyLoss()
loss = nn.CrossEntropyLoss()


# 创建图片数据迭代器
def colle(batch):
    # batch内多个元组形成一个元组，*解压出多个元组，zip每个对应位置缝合（相同索引）
    imgs, targets = list(zip(*batch))
    # 图片合并标签不合并可以加速训练（此处都合并了）
    imgs = torch.cat(imgs, dim=0)  # cat(inputs, dim=?)在给定维度上对输入的张量序列seq 进行连接操作。
    targets = torch.cat(targets, dim=0)  # tensor([1,]),tensor([0,])……（shape为[1,]）合并为tensor([[1],[0]])
    targets = targets.flatten()  # tensor([[1],[0]])二维转化为一维
    return imgs, targets


# 若实现了__len__和__getitem__，DataLoader会自动实现数据集的分批，shuffle打乱顺序，drop_last删除最后不完整的批次，collate_fn如何取样本
# dataload = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=colle)

# 创建可视化
def create_visualization(x_axis: list, y_axis: dict, type: Optional[str] = ['train', 'test']):
    plt.figure(figsize=(12, 6))
    lines = []  # 空列表保存线对象
    labels = list(y_axis.keys())
    for name in y_axis:
        line, = plt.plot(x_axis, y_axis[name].cpu().numpy())  # 逗号，返回单个线对象
        lines.append(line)
    plt.xlabel("Epochs")
    plt.legend(handles=lines, labels=labels, loc='best')

    # 设置坐标轴刻度
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax = plt.gca()  # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    plt.xlim(x_axis[0] - 1, x_axis[-1] + 1)  # 把x轴的刻度范围设置(不满一个刻度间隔时，数字不会显示出来，能看到一点空白)

    plt.grid(ls='--')  # 生成网格

    plt.savefig(f"{args.visualization_address}/{args.model}_{type}_result.png")
    plt.show()


# 创建logger
def creat_logger():
    logger = logging.getLogger(f"{args.model}_training")
    logger.setLevel(logging.INFO)
    # 创建一个handler，用于写入日志文件
    log_file = f"{args.log_address}/{args.model}_training.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def train():
    txt_log_file = open(f'{args.log_address}/{args.model}_training_log.txt', 'w')
    logger = creat_logger()

    global net
    epochs = args.epochs  # 设置epoch

    # 可视化参数
    total_train_losses = []  # 每个epoch的训练损失值列表
    accuracies = []  # 每个epoch的准确率列表
    precisions = []  # 每个epoch的精确率列表
    recalls = []  # 每个epoch的召回率列表
    f1_scores = []  # 每个epoch的F1值列表

    for epoch in range(epochs):
        start_epoch = time.time()  # epoch开始计时
        # 切分训练集和测试集
        trainset, testset = random_split(dataset, lengths=[args.train_rate, 1 - args.train_rate],
                                         generator=torch.Generator().manual_seed(0))
        trainLoader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=colle)
        testLoader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=colle)

        # 开启训练模式（BatchNorm和DropOut被使用，net.eval()推理模式会屏蔽这些模块）
        txt_log_file.write(f"Train_epoch:{epoch + 1}/{epochs}\n")
        txt_log_file.flush()
        logger.info(f"Train_epoch:{epoch + 1}/{epochs}")

        net.train()
        total_train_loss = 0
        batch_count = 0  # 对batch计数
        batch_counts = ceil(len(dataset) * args.train_rate / args.batch_size)
        for batch, (imgs, targets) in enumerate(trainLoader):
            start_batch = time.time()  # batch开始计时
            batch_count += 1
            # 训练主体
            # alexnet, vgg, resnet
            pred = net(imgs)  # imgs大小(batch_size,3,224,224)
            targets = targets.long()  # cross_entropy损失函数要求目标targets是长整型（torch.long或torch.int64）（都使用.long()）
            Loss = loss(pred, targets)
            total_train_loss += Loss
            optimizer.zero_grad()  # 优化器梯度归零
            Loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新参数
            # # GoogLeNet
            # pred, aux2, aux1 = net(imgs)
            # main_loss = loss(pred, targets)
            # # 计算辅助输出的损失
            # aux2_loss = loss(aux2.view(aux2.size(0), -1), targets)
            # aux1_loss = loss(aux1.view(aux1.size(0), -1), targets)
            # Loss = main_loss + 0.3 * aux2_loss + 0.3 * aux1_loss
            # total_loss += Loss
            # optimizer.zero_grad()
            # Loss.backward()
            # optimizer.step()

            # if Loss <= 1e-3:
            #     # loss达到理想值，提前终止
            #     torch.save(net.state_dict(), "./weights/A_Early_stop_epoch{}_params.pth".format(epoch + 1))
            #     return print("训练结束")

            # 打印参数
            batch_time = time.time() - start_batch  # 训练一个epoch的时间
            txt_log_file.write(f"batch:{batch_count}/{batch_counts}, "
                               f"loss:{float(Loss):.4f}, "
                               f"batch_time:{batch_time:.4f}\n")
            txt_log_file.flush()
            logger.info(f"batch:{batch_count}/{batch_counts}, "
                        f"loss:{float(Loss):.4f}, "
                        f"batch_time:{batch_time:.4f}")

        txt_log_file.write(f'Epoch:{epoch + 1}/{epochs}, total_loss:{float(total_train_loss):.4f}\n')
        txt_log_file.flush()
        logger.info(f'Epoch:{epoch + 1}/{epochs}, total_loss:{float(total_train_loss):.4f}')

        # 每个epoch保存一次参数
        torch.save(net.state_dict(), f"{args.weights_address}/{args.model}_epoch{epoch + 1}_params.pth")

        txt_log_file.write(f"Test_epoch:{epoch + 1}/{epochs}\n")
        txt_log_file.flush()
        logger.info(f"Test_epoch:{epoch + 1}/{epochs}")

        net.eval()
        preds = []  # 从模型获得预测结果
        true_labels = []  # 真实标签
        test_start_time = time.time()
        for batch, (imgs, targets) in enumerate(testLoader):
            # 测试主体
            pred = net(imgs).detach().cpu().numpy()  # 预测结果
            target = targets.ravel().cpu().numpy()
            preds.append(pred)
            true_labels.append(target)
        tensor_preds = [torch.from_numpy(pred) for pred in preds]
        preds = torch.cat(tensor_preds, dim=0)
        tensor_true_labels = [torch.from_numpy(label) for label in true_labels]
        true_labels = torch.cat(tensor_true_labels, dim=0)

        accuracy = accuracy_score(true_labels, preds.argmax(dim=1))
        precision = precision_score(true_labels, preds.argmax(dim=1), average='macro')
        recall = recall_score(true_labels, preds.argmax(dim=1), average='macro')
        f1 = f1_score(true_labels, preds.argmax(dim=1), average='macro')

        test_time = time.time() - test_start_time
        txt_log_file.write(f'Test Accuracy: {accuracy:.4f}, '
                           f'Test Precision: {precision:.4f}, '
                           f'Test Recall: {recall:.4f}, '
                           f'Test F1: {f1:.4f}, '
                           f'Test time: {test_time}\n')
        txt_log_file.flush()  # 确保内容被写入文件
        logger.info(f'Test Accuracy: {accuracy:.4f}, '
                    f'Test Precision: {precision:.4f}, '
                    f'Test Recall: {recall:.4f}, '
                    f'Test F1: {f1:.4f}, '
                    f'Test time: {test_time}')

        # 打印参数
        total_epoch_time = time.time() - start_epoch  # 训练一个epoch的时间
        epoch_hour = int(total_epoch_time / 60 // 60)
        epoch_minute = int(total_epoch_time // 60 - epoch_hour * 60)
        epoch_second = int(total_epoch_time - epoch_hour * 60 * 60 - epoch_minute * 60)
        txt_log_file.write(f"epoch:{epoch + 1}/{epochs}, "
                           f"total_time:{epoch_hour}:{epoch_minute}:{epoch_second}\n")
        txt_log_file.flush()
        logger.info(f"epoch:{epoch + 1}/{epochs}, "
                    f"total_time:{epoch_hour}:{epoch_minute}:{epoch_second}")

        total_train_losses.append(total_train_loss)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    y_train_axit = {"total_train_losses": total_train_losses, }
    y_test_axit = {"accuracies": accuracies,
                   "precisions": precisions,
                   "recalls": recalls,
                   "f1_scores": f1_scores}
    create_visualization(x_axis=list(range(epochs)), y_axis=y_train_axit, type='train')
    create_visualization(x_axis=list(range(epochs)), y_axis=y_test_axit, type='test')

    print("训练结束")
    txt_log_file.close()


if __name__ == '__main__':
    train()
