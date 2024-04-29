import os
import shutil


def segmentation_dataset(train_rate: float, train_address: str, test_address: str):
    """
    分割原始训练集
    :param train_rate: 训练集切分比例
    :param train_address: 训练集地址
    :param test_address: 测试集地址
    :return:
    """
    # 确保测试集地址存在
    if not os.path.exists(test_address):
        os.makedirs(test_address)
    if not os.path.exists(f"{test_address}/img"):
        os.makedirs(f"{test_address}/img")
    if not os.path.exists(f"{test_address}/label"):
        os.makedirs(f"{test_address}/label")

    # 获取原始训练集地址中的所有文件(7680张,5100张)
    for i in range(int(7680 * train_rate) + 1, 7680 + 1):
        shutil.move(f"{train_address}/img/{i}.jpg", f"{test_address}/img/{i}.jpg")
        shutil.move(f"{train_address}/label/{i}.txt", f"{test_address}/label/{i}.txt")
    for i in range(int(5100 * train_rate) + 1, 5100 + 1):
        shutil.move(f"{train_address}/img/red_{i}.jpg", f"{test_address}/img/red_{i}.jpg")
        shutil.move(f"{train_address}/label/red_{i}.txt", f"{test_address}/label/red_{i}.txt")


if __name__ == '__main__':
    # 训练集切分比例
    train_rate = 0.9
    # 原始训练集集地址
    train_address = "./train"
    # 目标测试集地址
    test_address = "./test"

    segmentation_dataset(train_rate=train_rate, train_address=train_address, test_address=test_address)

    # 打印完成信息
    print("File transfer completed.")
