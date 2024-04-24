import os
import shutil

# 训练集切分比例
train_rate = 0.9
# 原始训练集集地址
train_address = "./train"
# 目标测试集地址
test_address = "./test"

# 测试集地址中的所有文件合并到训练集中
for i in range(int(7680 * train_rate) + 1, 7680 + 1):
    shutil.move(f"{test_address}/img/{i}.jpg", f"{train_address}/img/{i}.jpg")
    shutil.move(f"{test_address}/label/{i}.txt", f"{train_address}/label/{i}.txt")
for i in range(int(5100 * train_rate) + 1, 5100 + 1):
    shutil.move(f"{test_address}/img/red_{i}.jpg", f"{train_address}/img/red_{i}.jpg")
    shutil.move(f"{test_address}/label/red_{i}.txt", f"{train_address}/label/red_{i}.txt")

# 打印完成信息
print("File transfer completed.")
