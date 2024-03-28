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
    net = torchvision.models.AlexNet(**kwargs)

    # print(net)
    x = torch.rand(size=(8, 3, 224, 224), device=device)
    from torchsummary import summary

    # 统计需要参数量
    # resout = summary(net, (3, 224, 224))
    # out = net(x)
    # print(out.size())
    # print(out)

    str = ["alexnet",
           "vgg11", "vgg13", "vgg16", "vgg19", "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
           "googlenet",
           "resnet18", "resnet34",
           "densenet121"
           "convnext_tiny", "convnext_small"]
    net = torchvision.models.convnext_tiny()

    net = torchvision.models.googlenet(**kwargs)
    net.to(device=device)
    # net.eval()
    outputs = net(x)
    print(outputs)  # 这将打印出返回的所有内容
    print(len(outputs))  # 这将打印出返回值的数量

    # resout = summary(net, (3, 224, 224))
