import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# 训练过程绘图出现错误，根据日志文件进行绘图
def parse_args():
    """
    解析命令行参数
    :return: args
    """
    parser = argparse.ArgumentParser(description='LogToVisualization')
    parser.add_argument('--model', default="googlenet", help='model')
    parser.add_argument('--log_address', default="./log", help='log_address')
    parser.add_argument('--visualization_address', default="./visualization", help='visualization_address')
    parser.add_argument('--epochs', default=20, type=int, help='epochs')
    return parser.parse_args()


def read_log_file(log_file):
    """
    读取日志文件
    :param log_file: 日志文件地址
    :return: total_train_losses, accuracies, precisions, recalls, f1_scores
    """
    total_train_losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    with open(log_file, 'r') as file:
        for line in file:
            line_list = line.strip().split(',')
            if len(line_list) == 2:
                epoch_str = line_list[1].split(':')
                if epoch_str[0] == " total_loss":
                    total_train_losses.append(float(epoch_str[1]))
            elif len(line_list) == 5:
                accuracies.append(float(line_list[0].split(':')[1]))
                precisions.append(float(line_list[1].split(':')[1]))
                recalls.append(float(line_list[2].split(':')[1]))
                f1_scores.append(float(line_list[3].split(':')[1]))
    return total_train_losses, accuracies, precisions, recalls, f1_scores


def create_visualization(x_axis, y_axis, visualization_address, model, chart_type):
    """
    绘制图表
    :param x_axis: x轴数据
    :param y_axis: y轴数据
    :param visualization_address: 可视化结果保存地址
    :param model: 模型名称
    :param chart_type: 图表类型，train or test
    :return:
    """
    plt.figure(figsize=(12, 6))
    lines = []
    labels = list(y_axis.keys())
    for name in y_axis:
        line, = plt.plot(x_axis, y_axis[name])
        lines.append(line)
    plt.xlabel("Epochs")
    plt.legend(handles=lines, labels=labels, loc='best')

    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(x_axis[0] - 1, x_axis[-1] + 1)
    plt.grid(ls='--')

    plt.savefig(f"{visualization_address}/{model}_{chart_type}_result.png")


if __name__ == '__main__':
    args = parse_args()
    log_file = os.path.join(args.log_address, f"{args.model}_training_log.txt")
    if not os.path.exists(log_file):
        print(f"Error: Log file {log_file} not found.")
    else:
        total_train_losses, accuracies, precisions, recalls, f1_scores = read_log_file(log_file)

        y_train_axis = {"total_train_losses": total_train_losses}
        y_test_axis = {
            "accuracies": accuracies,
            "precisions": precisions,
            "recalls": recalls,
            "f1_scores": f1_scores
        }

        create_visualization(list(range(1, len(total_train_losses) + 1)), y_train_axis, args.visualization_address,
                             args.model, 'train')
        create_visualization(list(range(1, len(accuracies) + 1)), y_test_axis, args.visualization_address, args.model,
                             'test')

        print(f"Log file {log_file} has been read and visualization has been created.")
