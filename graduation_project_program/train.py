from math import ceil

import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.myloss import MyLoss
from dataset.read_yolo_dataset import ReadYOLO
from Augmentation.data_augment import DataAugment

import logging
import argparse  # 可以直接在命令行中向程序传入参数并让程序运行
import time

# 在命令行运行时可以加以下参数进行修改
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--model', default="convnext_tiny", help='model')  # 选择模型
# 所有地址相关变量放在一个文件中，方便上云管理
parser.add_argument('--dateset_address', default="./dataset", help='dateset_address')  # 数据集地址
parser.add_argument('--train_rate', default=0.8, type=float, help='train_rate')  # 训练集切分比例

parser.add_argument('--lr', default=0.001, type=float, help='learning rate of model')  # 学习率
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')  # 动量
# （若上次的momentum(v)与此次的负梯度方向相同，则下降幅度加大，加速收敛）  v = momentum * v - learning_rate * d(weight); weight = weight + v
parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
parser.add_argument('--epochs', default=20, type=int, help='epochs')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')  # SGD的权重衰减

args = parser.parse_args()

# 创建全局device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取数据集
size = (224, 224)
data_augment = DataAugment(size=size)  # 数据增强实例化
dataset = ReadYOLO(dateset_address=args.dateset_address, phase='train', trans=data_augment, device=device)  # 读取数据集实例化
picture_num = len(dataset)  # 获取图片总数

kwargs = {"num_classes": 2}

# 模型实例化
net = torchvision.models.vgg16()
# alexnet vgg11/13/16/19(_bn) googlenet resnet18/34/50/101/152 densenet121/161/169/201 convnext_tiny/small/base/large
model_str = f"net = torchvision.models.{args.model}(**{kwargs})"
exec(model_str)
net = net.to(device=device)

# 迭代器和损失函数优化器实例化
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
loss = MyLoss()  # 等价于loss = nn.CrossEntropyLoss()，loss(网络获取的图片)


# 创建图片数据迭代器
def colle(batch):
    # batch内多个元组形成一个元组，*解压出多个元组，zip每个对应位置缝合（相同索引）
    imgs, targets = list(zip(*batch))
    # 图片合并标签不合并可以加速训练（此处都合并了）
    imgs = torch.cat(imgs, dim=0)  # cat(inputs, dim=?)在给定维度上对输入的张量序列seq 进行连接操作。
    targets = torch.cat(targets, dim=0)  # tensor([1,]),tensor([0,])……（shape为[1,]）合并为tensor([1,0])
    return imgs, targets


# 若实现了__len__和__getitem__，DataLoader会自动实现数据集的分批，shuffle打乱顺序，drop_last删除最后不完整的批次，collate_fn如何取样本
dataload = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=colle)


def train():
    log_file = open(f'./log/{args.model}_training_log.txt', 'w')

    global net
    epochs = args.epochs  # 设置epoch

    for epoch in range(epochs):
        start_epoch = time.time()  # epoch开始计时
        # 切分训练集和测试集
        trainset, testset = random_split(dataset, lengths=[args.train_rate, 1 - args.train_rate],
                                         generator=torch.Generator().manual_seed(0))
        trainLoader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=colle)
        testLoader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=colle)

        # 开启训练模式（BatchNorm和DropOut被使用，net.eval()推理模式会屏蔽这些模块）
        log_file.write(f"Train_epoch:{epoch + 1}/{epochs}\n")
        log_file.flush()
        print(f"Train_epoch:{epoch + 1}/{epochs}\n")

        net.train()
        Loss = 0
        total_loss = 0
        batch_count = 0  # 对batch计数
        batch_counts = ceil(len(dataset) * args.train_rate / args.batch_size)
        for batch, (imgs, targets) in enumerate(trainLoader):
            start_batch = time.time()  # batch开始计时
            batch_count += 1
            # 训练主体
            # alexnet, vgg, resnet
            pred = net(imgs)  # imgs大小(batch_size,3,224,224)
            Loss = loss(pred, targets)
            total_loss += Loss
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
            log_file.write(f"batch:{batch_count}/{batch_counts}, "
                           f"loss:{float(Loss):.4f}, "
                           f"batch_time:{batch_time:.4f}\n")
            log_file.flush()
            print(f"batch:{batch_count}/{batch_counts}, "
                  f"loss:{float(Loss):.4f}, "
                  f"batch_time:{batch_time:.4f}\n")

        log_file.write(f'Epoch {epoch + 1}/{epochs}, total_loss: {float(total_loss):.4f}\n')
        log_file.flush()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {float(total_loss):.4f}\n')

        torch.save(net.state_dict(), f"./weights/{args.model}_epoch{epoch + 1}_params.pth")  # 每个epoch保存一次参数

        log_file.write(f"Test_epoch:{epoch + 1}/{epochs}\n")
        log_file.flush()
        print(f"Test_epoch:{epoch + 1}/{epochs}\n")

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
        log_file.write(f'Test Accuracy: {accuracy:.4f}, '
                       f'Test Precision: {precision:.4f}, '
                       f'Test Recall: {recall:.4f}, '
                       f'Test F1: {f1:.4f} '
                       f'Test time: {test_time}\n')
        log_file.flush()  # 确保内容被写入文件
        print(f'Test Accuracy: {accuracy:.4f}, '
              f'Test Precision: {precision:.4f}, '
              f'Test Recall: {recall:.4f}, '
              f'Test F1: {f1:.4f} '
              f'Test time: {test_time}\n')

        # 打印参数
        total_epoch_time = time.time() - start_epoch  # 训练一个epoch的时间
        epoch_hour = int(total_epoch_time / 60 // 60)
        epoch_minute = int(total_epoch_time // 60 - epoch_hour * 60)
        epoch_second = int(total_epoch_time - epoch_hour * 60 * 60 - epoch_minute * 60)
        log_file.write(f"epoch:{epoch + 1}/{epochs}, "
                       f"total_time:{epoch_hour}:{epoch_minute}:{epoch_second}\n")
        log_file.flush()
        print(f"epoch:{epoch + 1}/{epochs}, "
              f"total_time:{epoch_hour}:{epoch_minute}:{epoch_second}\n")

    print("训练结束")

    log_file.close()


if __name__ == '__main__':
    train()
