import torchvision
from torch import nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

kwargs = {"num_classes": 2}
# , "init_weights": True, "dropout": 0.5}
# vgg16 = torchvision.models.vgg16  # torchvision提供网络
# vgg16.classifier[6] = nn.Linear(in_features=4096, out_features=2)
# vgg16.to(device=device)
# print(vgg16)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = torchvision.models.alexnet(**kwargs)
    net.to(device=device)

    # print(net)
    from torchsummary import summary

    # from torchinfo import summary

    # 统计需要参数量
    summary(net, (3, 224, 224))
    # out = net(x)
    # print(out.size())
    # print(out)

    str = ["alexnet",
           "vgg11", "vgg13", "vgg16", "vgg19", "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
           "googlenet",
           "resnet18", "resnet34",
           "densenet121"
           "convnext_tiny", "convnext_small"]
    # net = torchvision.models.convnext_tiny()

    # net = torchvision.models.googlenet(**kwargs)
    # net.to(device=device)
    # # net.eval()
    # outputs = net(x)
    # print(outputs)  # 这将打印出返回的所有内容
    # print(len(outputs))  # 这将打印出返回值的数量

    # resout = summary(net, (3, 224, 224))

# import torch
# import torchvision
# from torchsummary import summary
#
# kwargs = {"num_classes": 2}
#
# # 实例化您的模型
# model = torchvision.models.densenet121(**kwargs)
# model.eval()  # 将模型设置为评估模式
#
# # 定义输入尺寸（不包括批处理大小）
# input_size = (3, 224, 224)  # 例如，对于RGB图像，高度和宽度为224x224
#
# # 创建一个包含输入尺寸的列表（如果模型有多个输入，请为每个输入添加相应的尺寸）
# input_sizes = [input_size]  # 这是一个单输入的示例，如果是多输入，请添加更多尺寸
#
# # 生成随机输入张量
# x = [torch.rand(2, *in_size) for in_size in input_sizes]  # 批处理大小为2
#
# # 生成模型摘要
# summary(model, x)
