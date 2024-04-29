import os
import shutil


def merge_dataset(train_rate: float, train_address: str, test_address: str):
    """
    合并训练集和测试集
    :param train_rate: 训练集切分比例
    :param train_address: 训练集地址
    :param test_address: 测试集地址
    :return:
    """
    # 测试集地址中的所有文件合并到训练集中
    for i in range(int(7680 * train_rate) + 1, 7680 + 1):
        shutil.move(f"{test_address}/img/{i}.jpg", f"{train_address}/img/{i}.jpg")
        shutil.move(f"{test_address}/label/{i}.txt", f"{train_address}/label/{i}.txt")
    for i in range(int(5100 * train_rate) + 1, 5100 + 1):
        shutil.move(f"{test_address}/img/red_{i}.jpg", f"{train_address}/img/red_{i}.jpg")
        shutil.move(f"{test_address}/label/red_{i}.txt", f"{train_address}/label/red_{i}.txt")


if __name__ == '__main__':
    # 训练集切分比例
    train_rate = 0.9
    # 原始训练集集地址
    train_address = "./train"
    # 目标测试集地址
    test_address = "./test"

    merge_dataset(train_rate, train_address, test_address)

    # 打印完成信息
    print("File transfer completed.")
