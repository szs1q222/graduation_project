import torchvision
import cv2
import torch
import argparse
from PIL import Image


def get_args():
    """
    获取命令行参数
    :return: args
    """
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--model', default="vgg16", help='model')  # 选择模型
    parser.add_argument('--num_classes', default=2, type=int, help='num_classes')  # 目标分类类别数
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout of model')  # dropout
    parser.add_argument('--weight_dir', default='./weights/vgg16/vgg16_epoch12_params.pth', help="参数路径")
    parser.add_argument('--test_dir', default='./dataset/test/img/Image_1.jpg', help="测试图片路径")
    return parser.parse_args()


def run(args):
    """
    进行推理
    :param args: 命令行参数
    :return: 预测概率, 预测结果
    """
    try:
        # 加载模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        kwargs = {"num_classes": args.num_classes}  # 模型参数存储
        if args.model.lower() not in {'resnet18', 'resnet34', 'resnet50', 'densenet121', 'densenet201', 'densenet169',
                                      'densenet161'}:
            kwargs["dropout"] = args.dropout
        net = getattr(torchvision.models, args.model)(**kwargs)
        net.load_state_dict(torch.load(args.weight_dir, map_location=device))
        net = net.to(device=device)

        # 读取测试图片
        image = Image.open(args.test_dir).convert('RGB')
        trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
        ])
        image = trans(image).unsqueeze(0).to(device=device)

        # 推理
        net.eval()
        result = torch.argmax(net(image).ravel())
        return net(image).ravel(), result
    except Exception as e:
        print(f"Error occurred during inference: {e}")
        return None, None


if __name__ == '__main__':
    classification = {0: '非红包', 1: '红包'}
    args = get_args()
    probability, result = run(args)
    if probability is not None and result is not None:
        print(f"Probability: {probability}, Result: {result}")
        print(f"Predicted label: {classification[result.item()]}")
    else:
        print("Inference failed.")
