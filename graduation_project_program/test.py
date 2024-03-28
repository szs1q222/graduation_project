import torchvision
from PIL import Image
from torch import nn
from model_VGG16 import VGG16

import cv2
import torch
import argparse

parser = argparse.ArgumentParser(description='VGG16 Testing')
parser.add_argument('--weight_dir', default='./weights/VGG16_epoch1_params.pth', help="参数路径")
parser.add_argument('--test_dir', default='./dataset/test/img/Image_4.jpg', help="测试图片路径")

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 创建全局device

# 加载模型
net = VGG16(mode="test")
net = net.to(device=device)

# 加载模型参数
net.load_state_dict(torch.load(args.weight_dir))


# 进行推理
def run():
    net.eval()
    image = cv2.imread(args.test_dir, cv2.IMREAD_COLOR)
    trans = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor()])
    image = trans(image).unsqueeze(0).to(device=device)
    result = torch.argmax(net(image).ravel())  # ravel()方法将数组维度拉成一维数组,argmax选出最大值对应索引
    return net(image).ravel(), result


if __name__ == '__main__':
    probability, result = run()
    print(probability, result)
