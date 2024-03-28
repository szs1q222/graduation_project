import pandas as pd
import os

# 获取文件名与标签的映射表
red_packet_dir = r"./dataset/labels_my-project-name.csv"
red_packet = pd.read_csv(red_packet_dir, header=None)

# 第一列为名字，第二列为类别
red_packet_name = red_packet.iloc[:, 0]
red_packet_label = red_packet.iloc[:, 1]

# 填充label文件夹内容
for i, name in enumerate(red_packet_name):
    with open(os.path.join(r"E:\学习\毕设\数据处理文件\dataset\label", name.split('.')[0] + ".txt"), 'w') as f:
        f.write(red_packet_label[i].split('[')[1].split(']')[0])
