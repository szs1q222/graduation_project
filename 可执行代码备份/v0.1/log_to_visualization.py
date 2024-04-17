import argparse
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 训练过程绘图出现错误，根据日志文件进行绘图
parser = argparse.ArgumentParser(description='LogToVisualization')
parser.add_argument('--model', default="googlenet", help='model')  # 选择模型
parser.add_argument('--log_address', default="./log", help='log_address')  # 日志地址
parser.add_argument('--visualization_address', default="./visualization", help='visualization_address')  # 可视化存储地址
# 训练模型相关参数设置
parser.add_argument('--epochs', default=20, type=int, help='epochs')
args = parser.parse_args()

epochs = args.epochs
total_train_losses = []
accuracies = []
precisions = []
recalls = []
f1_scores = []

with open(f"{args.log_address}/{args.model}_training_log.txt", 'r') as file:
    for line in file:
        line_list = line.strip().split(',')
        if len(line_list) == 2:
            # Epoch 1 / 20, total_loss: 212.7100 或者 Epoch:1/20, total_time:0:9:44
            epoch_str = line_list[1].split(':')
            if epoch_str[0] == " total_loss":
                total_train_losses.append(float(epoch_str[1]))
        if len(line_list) == 5:
            # print(line_list)
            # Test Accuracy: 0.5914, Test Precision: 0.2957, Test Recall: 0.5000, Test F1: 0.3716 Test time: 112.91886448860168
            # Test Accuracy: 0.9777, Test Precision: 0.9788, Test Recall: 0.9751, Test F1: 0.9768, Test time: 119.29520273208618
            accuracies.append(float(line_list[0].split(':')[1]))
            precisions.append(float(line_list[1].split(':')[1]))
            recalls.append(float(line_list[2].split(':')[1]))
            f1_scores.append(float(line_list[3].split(':')[1]))

y_axis = {
    # "total_train_losses": total_train_losses,
    "accuracies": accuracies,
    "precisions": precisions,
    "recalls": recalls,
    "f1_scores": f1_scores}


# print(y_axis)

def create_visualization(x_axis: list, y_axis: dict, type: Optional[str] = ['train', 'test']):
    plt.figure(figsize=(12, 6))
    lines = []  # 空列表保存线对象
    labels = list(y_axis.keys())
    for name in y_axis:
        line, = plt.plot(x_axis, y_axis[name])  # 注意逗号，返回单个线对象
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


if __name__ == '__main__':
    y_train_axit = {"total_train_losses": total_train_losses, }
    y_test_axit = {"accuracies": accuracies,
                   "precisions": precisions,
                   "recalls": recalls,
                   "f1_scores": f1_scores}

    y_train_axit_len = len(y_train_axit["total_train_losses"])
    y_test_axit_len = len(y_test_axit["accuracies"])

    create_visualization(x_axis=list(range(1, y_train_axit_len + 1)), y_axis=y_train_axit, type='train')
    create_visualization(x_axis=list(range(1, y_test_axit_len + 1)), y_axis=y_test_axit, type='test')
