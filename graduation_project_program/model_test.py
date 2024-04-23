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

def replace_activation(model, old_activation_type, new_activation_type, new_activation_params):
    for name, module in model.named_children():
        if isinstance(module, old_activation_type):
            # 直接替换为新的激活函数
            setattr(model, name, new_activation_type(**new_activation_params))
        elif isinstance(module, nn.Sequential):
            # 递归地处理序列模块
            for child_name, child_module in module.named_children():
                if isinstance(child_module, old_activation_type):
                    setattr(module, child_name, new_activation_type(**new_activation_params))
        elif hasattr(module, 'children') and len(list(module.children())) > 0:
            # 递归地处理其他模块
            replace_activation(module, old_activation_type, new_activation_type, new_activation_params)
    return model


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = torchvision.models.alexnet(**kwargs)
    # net = torchvision.models.googlenet(**kwargs)
    # net = torchvision.models.convnext_tiny()
    # net = torchvision.models.vit_b_16(**kwargs)
    net.to(device=device)

    nn.CrossEntropyLoss()
    # net.eval()
    # x = torch.rand(2, 3, 224, 224).to(device)
    # outputs = net(x)
    # print(outputs)  # 这将打印出返回的所有内容
    # print(len(outputs))  # 这将打印出返回值的数量
    print(net)

    new_activation_params = {'negative_slope': 0.1}
    net = replace_activation(net, nn.ReLU, nn.LeakyReLU, new_activation_params)
    torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print(net)

    # 统计需要参数量
    # from torchsummary import summary
    #
    # summary(net, (3, 224, 224))

    str = ["alexnet",
           "vgg11", "vgg13", "vgg16", "vgg19", "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
           "googlenet",
           "resnet18", "resnet34",
           "densenet121"
           "convnext_tiny", "convnext_small"]

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
