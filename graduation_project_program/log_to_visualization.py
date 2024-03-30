import argparse

import matplotlib.pyplot as plt

# 训练过程绘图出现错误，根据日志文件进行绘图
parser = argparse.ArgumentParser(description='LogToVisualization')
parser.add_argument('--model', default="alexnet", help='model')  # 选择模型
parser.add_argument('--log_address', default="./log", help='log_address')  # 日志地址
parser.add_argument('--visualization_address', default="./visualization", help='visualization_address')  # 可视化存储地址
# 训练模型相关参数设置
parser.add_argument('--epochs', default=20, type=int, help='epochs')
args = parser.parse_args()

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
        if len(line_list) == 4:
            # Test Accuracy: 0.5914, Test Precision: 0.2957, Test Recall: 0.5000, Test F1: 0.3716 Test time: 112.91886448860168
            accuracies.append(float(line_list[0].split(':')[1]))
            precisions.append(float(line_list[1].split(':')[1]))
            recalls.append(float(line_list[2].split(':')[1]))
            f1_scores.append(float(line_list[3].split(' ')[3]))

y_axis = {
    # "total_train_losses": total_train_losses,
    "accuracies": accuracies,
    "precisions": precisions,
    "recalls": recalls,
    "f1_scores": f1_scores}
print(y_axis)

plt.figure(figsize=(12, 6))
lines = []  # 创建一个空列表来保存线对象
labels = list(y_axis.keys())  # 获取所有标签
for name in y_axis:
    line, = plt.plot(list(range(20)), y_axis[name])  # 注意这里的逗号，它会返回单个线对象
    lines.append(line)  # 将线对象添加到列表中
plt.xlabel("Epochs")
plt.legend(handles=lines, labels=labels, loc='best')  # 传递 handles 和 labels 参数
plt.savefig(f"{args.visualization_address}/{args.model}_training_result.png")
plt.show()
