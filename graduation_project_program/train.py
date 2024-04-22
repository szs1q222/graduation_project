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


def parse_arguments():
    """
    解析命令行参数
    :return: 命令行参数
    """
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--model', default="resnet50", help='model')  # 选择模型
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
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout of model')  # dropout
    parser.add_argument('--optimizer', default="SGD", help='optimizer')  # 优化器选择
    parser.add_argument('--loss_function', default="CrossEntropyLoss", help='loss function')  # 损失函数选择
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')  # 动量
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
    parser.add_argument('--epochs', default=20, type=int, help='epochs')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')  # SGD的权重衰减
    return parser.parse_args()


def prepare_folders(args):
    """
    创建存储文件夹
    :param args: 命令行参数
    :return:
    """
    for folder in [args.weights_address, args.log_address, args.visualization_address]:
        if not os.path.exists(folder):
            os.makedirs(folder)


def prepare_data(args):
    """
    读取数据集
    :param args: 命令行参数
    :return: 数据集实例化, device
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    size = (224, 224)
    data_augment = DataAugment(size=size)  # 数据增强实例化
    dataset = ReadYOLO(dateset_address=args.dateset_address, phase='train', trans=data_augment,
                       device=device)  # 读取数据集实例化
    # picture_num = len(dataset)  # 获取图片总数
    return dataset, device


def prepare_model(args, device):
    """
    模型实例化
    :param args: 命令行参数
    :param device:
    :return: 模型实例化, 优化器实例化, 损失函数实例化
    """
    kwargs = {"num_classes": args.num_classes}  # 模型参数存储
    if args.model.lower() not in {'resnet18', 'resnet34', 'resnet50', 'densenet121', 'densenet201', 'densenet169',
                                  'densenet161'}:
        kwargs["dropout"] = args.dropout

    # 动态导入模型
    net = getattr(torchvision.models, args.model.lower())(**kwargs)
    net = net.to(device=device)

    # 迭代器和损失函数优化器实例化
    optimizer = getattr(torch.optim, args.optimizer)(net.parameters(), lr=args.lr, momentum=args.momentum,
                                                     weight_decay=args.weight_decay)
    # loss = MyLoss()  # 等价于loss = nn.CrossEntropyLoss()
    loss = getattr(nn, args.loss_function)()
    return net, optimizer, loss


def colle(batch):
    """
    创建图片数据迭代器
    :param batch: 每个batch的数据
    :return: 图片数据迭代器, 标签数据迭代器
    """
    # batch内多个元组形成一个元组，*解压出多个元组，zip每个对应位置缝合（相同索引）
    imgs, targets = list(zip(*batch))
    # 图片合并标签不合并可以加速训练（此处都合并了）
    imgs = torch.cat(imgs, dim=0)  # cat(inputs, dim=?)在给定维度上对输入的张量序列seq 进行连接操作。
    targets = torch.cat(targets, dim=0)  # tensor([1,]),tensor([0,])……（shape为[1,]）合并为tensor([[1],[0]])
    targets = targets.flatten()  # tensor([[1],[0]])二维转化为一维
    return imgs, targets


# 若实现了__len__和__getitem__，DataLoader会自动实现数据集的分批，shuffle打乱顺序，drop_last删除最后不完整的批次，collate_fn如何取样本
# dataload = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=colle)

def create_visualization(args, x_axis: list, y_axis: dict, type: Optional[str] = ['train', 'test']):
    """
    创建可视化
    :param args: 命令行参数
    :param x_axis: x轴数据
    :param y_axis: y轴数据
    :param type: 参数来源类型, train/test
    :return:
    """
    plt.figure(figsize=(12, 6))
    for name, values in y_axis.items():
        plt.plot(x_axis, values, label=name)
    plt.xlabel("Epochs")
    plt.legend(loc='best')

    # 设置坐标轴刻度
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))  # 把x轴的刻度间隔设置为1
    plt.grid(ls='--')  # 生成网格
    plt.savefig(f"{args.visualization_address}/{args.model.lower()}_{type}_result.png")
    # plt.show()


# 创建logger
def creat_logger(args):
    """
    创建logger(日志记录)
    :param args: 命令行参数
    :return: logger实例化
    """
    logger = logging.getLogger(f"{args.model.lower()}_training")
    logger.setLevel(logging.INFO)
    log_file = f"{args.log_address}/{args.model.lower()}_training.log"
    # 创建一个handler，用于写入日志文件
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


def calculate_metrics(true_labels, preds):
    """
    计算准确率,精确率,召回率,F1值
    :param true_labels: 真实标签
    :param preds: 预测标签
    :return: 准确率,精确率,召回率,F1值
    """
    accuracy = accuracy_score(true_labels, preds.argmax(dim=1))
    precision = precision_score(true_labels, preds.argmax(dim=1), average='macro')
    recall = recall_score(true_labels, preds.argmax(dim=1), average='macro')
    f1 = f1_score(true_labels, preds.argmax(dim=1), average='macro')
    return accuracy, precision, recall, f1


def train(args):
    """
    模型训练
    :param args: 命令行参数
    :return:
    """
    prepare_folders(args)

    txt_log_file = open(f'{args.log_address}/{args.model.lower()}_training_log.txt', 'w')
    logger = creat_logger(args)

    dataset, device = prepare_data(args)
    net, optimizer, loss = prepare_model(args, device)

    epochs = args.epochs  # 设置epoch

    # 可视化参数
    total_train_losses = []  # 每个epoch的训练损失值列表
    accuracies, precisions, recalls, f1_scores = [], [], [], []  # 每个epoch的准确率,精确率,召回率,F1值列表

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
            # targets = targets.long()
            # main_loss = loss(pred, targets)
            # # 计算辅助输出的损失
            # aux2_loss = loss(aux2.view(aux2.size(0), -1), targets)
            # aux1_loss = loss(aux1.view(aux1.size(0), -1), targets)
            # Loss = main_loss + 0.3 * aux2_loss + 0.3 * aux1_loss
            # total_train_loss += Loss
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

        # 打印训练集总损失
        txt_log_file.write(f'Epoch:{epoch + 1}/{epochs}, total_loss:{float(total_train_loss):.4f}\n')
        txt_log_file.flush()
        logger.info(f'Epoch:{epoch + 1}/{epochs}, total_loss:{float(total_train_loss):.4f}')

        # 每个epoch保存一次参数
        torch.save(net.state_dict(), f"{args.weights_address}/{args.model.lower()}_epoch{epoch + 1}_params.pth")

        # 删除上一个epoch的模型参数文件
        if epoch > 0:
            previous_epoch_params = f"{args.weights_address}/{args.model.lower()}_epoch{epoch}_params.pth"
            if os.path.exists(previous_epoch_params):
                os.remove(previous_epoch_params)

        # 开始测试
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

        # 处理预测结果和真实标签
        tensor_preds = [torch.from_numpy(pred) for pred in preds]
        preds = torch.cat(tensor_preds, dim=0)
        tensor_true_labels = [torch.from_numpy(label) for label in true_labels]
        true_labels = torch.cat(tensor_true_labels, dim=0)

        # 计算准确率,精确率,召回率,F1值
        accuracy, precision, recall, f1 = calculate_metrics(true_labels, preds)

        test_time = time.time() - test_start_time
        # 打印测试集准确率,精确率,召回率,F1值
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

        total_epoch_time = time.time() - start_epoch  # 训练一个epoch的时间
        epoch_hour = int(total_epoch_time / 60 // 60)
        epoch_minute = int(total_epoch_time // 60 - epoch_hour * 60)
        epoch_second = int(total_epoch_time - epoch_hour * 60 * 60 - epoch_minute * 60)
        # 打印训练一个epoch的时间
        txt_log_file.write(f"epoch:{epoch + 1}/{epochs}, "
                           f"total_time:{epoch_hour}:{epoch_minute}:{epoch_second}\n")
        txt_log_file.flush()
        logger.info(f"epoch:{epoch + 1}/{epochs}, "
                    f"total_time:{epoch_hour}:{epoch_minute}:{epoch_second}")

        total_train_loss = total_train_loss.cpu().detach().numpy()
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
    create_visualization(args=args, x_axis=list(range(epochs)), y_axis=y_train_axit, type='train')
    create_visualization(args=args, x_axis=list(range(epochs)), y_axis=y_test_axit, type='test')

    print("训练结束")
    txt_log_file.close()


def main():
    args = parse_arguments()
    train(args)


if __name__ == '__main__':
    main()
