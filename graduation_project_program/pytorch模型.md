1. **AlexNet** - 2012年ImageNet竞赛冠军网络，由Alex Krizhevsky等人设计。它是第一个在ImageNet数据集上取得突破性效果的深度卷积神经网络，使用了ReLU激活函数和Dropout来防止过拟合。

   alexnet

2. **ConvNext** - 一个纯粹的卷积神经网络，它基于一个简单的宏观设计准则，即扩大网络规模，逐步改进每层的设计，从而实现了在多个任务上的最先进性能。

   convnext_tiny
   convnext_small
   convnext_base
   convnext_large

3. **DenseNet** - 通过将每层与其他层连接，而不是只与前一层连接，减少了参数的数量，并增加了梯度回传的路径，有助于训练更深层的网络。

   densenet121  
   densenet161
   densenet169
   densenet201

4. **EfficientNet** - 一个自动化的网络架构搜索方法，它通过复合缩放方法来均匀地放大网络深度、宽度和分辨率，以获得更好的效率和准确性。

   efficientnet_b0
   efficientnet_b1
   efficientnet_b2
   efficientnet_b3
   efficientnet_b4
   efficientnet_b5
   efficientnet_b6
   efficientnet_b7

5. **EfficientNetV2** - EfficientNet的改进版，它通过优化网络架构和训练方法，进一步提高了效率和准确性。

   efficientnet_v2_s 
   efficientnet_v2_m 
   efficientnet_v2_l

6. **GoogLeNet/Inception** - 通过使用不同尺寸的卷积核和池化层来同时捕获图像的局部和全局特征，这有助于网络更有效地学习。

   googlenet

7. **Inception V3** - Inception模型的第三版，它通过改进网络结构和训练方法，提高了模型的准确性和效率。

   inception_v3

8. **MaxVit** - 一种新颖的视觉变换器架构，它通过引入最大池化层来提高效率和准确性。

   maxvit_t

9. **MNASNet** - 一种使用多目标优化的移动设备神经网络架构搜索方法，旨在找到在给定计算预算下具有最佳准确性和效率的网络架构。

   mnasnet0_5
   mnasnet0_75
   mnasnet1_0
   mnasnet1_3

10. **MobileNet V2** - 通过引入线性瓶颈和倒置残差块来减少参数数量和计算成本，同时保持高准确性。

    mobilenet_v2

11. **MobileNet V3** - 进一步优化MobileNet V2，通过引入新的架构改进和神经架构搜索技术，提高了效率和准确性。

    mobilenet_v3_large
    mobilenet_v3_small

12. **RegNet** - 一种基于规则的神经网络架构，它通过系统地调整网络设计参数来探索设计空间，以找到具有良好性能和效率的网络。

    regnet_y_400mf
    regnet_y_800mf
    regnet_y_1_6gf
    regnet_y_3_2gf
    regnet_y_8gf
    regnet_y_16g
    regnet_y_32gf
    regnet_y_128gf
    regnet_x_400mf
    regnet_x_800mf
    regnet_x_1_6gf
    regnet_x_3_2gf
    regnet_x_8gf
    regnet_x_16gf
    regnet_x_32gf 

13. **ResNet** - 残差网络通过引入残差块来解决深层网络的训练问题，允许网络进行更深的训练而不会出现性能下降。

    resnet18
    resnet34
    resnet50
    resnet101
    resnet152

14. **ResNeXt** - ResNet的变体，它通过使用分组卷积和增加基数（而不是深度或宽度）来提高准确性。

    resnext50_32x4d
    resnext101_32x8d
    resnext101_64x4d

15. **ShuffleNet V2** - 通过引入通道重排操作和改进的网络架构来减少计算成本，同时保持高准确性。

    shufflenet_v2_x0_5
    shufflenet_v2_x1_0
    shufflenet_v2_x1_5
    shufflenet_v2_x2_0

16. **SqueezeNet** - 通过使用fire模块（包括squeeze和expand操作）来减少参数数量和计算成本，同时保持高准确性。

    squeezenet1_0
    squeezenet1_1

17. **Swin Transformer** - 一种基于Transformer的视觉模型，它通过使用层次结构和平移等变性的窗口划分来捕获图像的局部和全局特征。

    swin_t
    swin_s
    swin_b
    swin_v2_t
    swin_v2_s
    swin_v2_b

18. **VGG** - 一种使用多个3x3卷积层的深度卷积神经网络，它在ImageNet竞赛中取得了很好的效果，但参数数量和计算成本较高。

    vgg11
    vgg11_bn
    vgg13
    vgg13_bn
    vgg16
    vgg16_bn
    vgg19
    vgg19_bn

19. **Vision Transformer** - 是一种将Transformer架构应用于图像识别任务的模型，它通过将图像分割成多个块并使用自注意力机制来捕获图像特征。

    vit_b_16
    vit_b_32
    vit_l_16
    vit_l_32
    vit_h_14

20. **Wide ResNet** - 通过增加网络的宽度（而不是深度）来提高准确性，它使用更宽的残差块来捕获更多的特征。

    wide_resnet50_2
    wide_resnet101_2

- alexnet：有5个卷积层，3个最大池化层，2个归一化层，2个全连接层和1个分类层。
- convnext_tiny：具有较少的卷积层和深度可分离卷积，旨在减小模型大小和提高计算效率。
- convnext_small：与convnext_tiny类似，但更深一些，拥有更多的卷积层。
- convnext_base：基础版本的ConvNeXt模型，有更多的卷积层和参数。
- convnext_large：大版本的ConvNeXt模型，参数更多，卷积层更深。
- densenet121：具有121层，其中每个层以前馈方式连接到其他每一层。
- densenet161：比densenet121更深，具有161层。
- densenet169：具有169层，较densenet121更深。
- densenet201：具有201层，是densenet系列中最深的模型之一。
- efficientnet_b0：EfficientNet的基准模型，具有较小的深度和宽度。
- efficientnet_b1-b7：EfficientNet的变体，随着数字的增加，模型的深度和宽度逐渐增加。
- efficientnet_v2_s/m/l：EfficientNet的第二个版本，提供小、中、大三种规模的模型。
- googlenet：有9个卷积层，其中包括多个辅助分类器和一个主分类器。
- inception_v3：具有48层，使用了Inception模块和辅助分类器。
- maxvit_t：MaxViT的较小版本，结合了Transformer和CNN的优点。
- mnasnet0_5-1_3：一系列模型，具有不同的深度和宽度，旨在优化移动设备上的速度和准确性。
- mobilenet_v2：具有29层，使用了反转残差块。
- mobilenet_v3_large/small：第三版MobileNet，提供大、小两种规模，使用了新颖的块结构。
- regnet_y/x_400mf-128gf：RegNet系列模型，提供不同规模和性能的模型。
- resnet18-152：具有18到152层，使用残差块。
- resnext50_32x4d：具有50层，使用分组卷积和残差连接。
- resnext101_32x8d：具有101层，使用更深的层和更多的分组。
- resnext101_64x4d：与resnext101_32x8d类似，但分组数更多。
- shufflenet_v2_x0_5-x2_0：一系列模型，旨在优化移动设备上的速度和准确性。
- squeezenet1_0/1_1：具有较少的参数和计算量，使用了Squeeze和Excitation模块。
- swin_t/s/b：Swin Transformer的变体，提供小、中、大三种规模的模型。
- swin_v2_t/s/b：Swin Transformer的第二版，提供小、中、大三种规模的模型。
- vgg11-19_bn：具有11到19层，使用多个卷积层和池化层。
- vit_b_16/32：Vision Transformer的基准模型，具有不同的图像 patch 大小。
- vit_l_16/32：比vit_b更大，具有更多的层和参数。
- vit_h_14：Vision Transformer的超大型版本，具有大量的层和参数。
- wide_resnet50_2：具有50层，使用了更宽的层。
- wide_resnet101_2：具有101层，使用了更宽的层。