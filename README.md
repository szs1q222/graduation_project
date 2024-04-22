# Graduation_Project
基于深度学习的移动应用红包识别技术研究
Research on red packet recognition technology of mobile application based on deep learning

## 1. 项目介绍 Project Introduction

| 条目 Item       | 详情 Detail                      |
| --------------- | -------------------------------- |
| 类型 Type       | 毕业设计 graduation project      |
| 学校 School     | 东南大学 Southeast University    |
| 专业 Major      | 人工智能 Artificial Intelligence |
| 学历 Education  | 本科 Undergraduate               |
| 学号 Student ID | 58120120                         |
| 姓名 Name       | 石知晟 Shi Zhisheng              |
| 导师 Supervisor | 戚晓芳 Qi Xiaofang               |

### 1.1 项目背景 Project Context

​		近年来，在各种移动应用中都出现了各种网络红包的应用，但其中也滋生出了红包欺诈等恶意的行为。深度学习是当今图像识别技术领域中非常关键的技术手段之一，应用前景广阔，其在人工智能发展领域以及视觉应用领域都具有积极的现实意义。因此，通过深度学习技术来识别和检测移动应用中的红包图像，对于预防和打击红包欺诈行为具有重要的现实意义。

​		In recent years, a variety of online red envelope applications have appeared in various mobile applications, but there are also malicious behaviors such as red envelope fraud. Deep learning is one of the key technical means in the field of image recognition technology, which has a broad application prospect, and has positive practical significance in the field of artificial intelligence development and vision application. Therefore, it is of great practical significance to identify and detect red envelope images in mobile applications through deep learning technology to prevent and combat red envelope fraud.

### 1.2 内容和目标 Contents and Objectives

​		围绕与深度学习相关的图像识别方法，检测和识别移动应用中的红包图像，为预防和打击红包欺诈行为的研究提供支持。通过了解现有经典方法，在现有方法上寻找解决思路并进行实验，以挖掘更多适合此任务的解决方案，并提升模型性能。

​		Around the image recognition method related to deep learning, the detection and recognition of red envelope images in mobile applications will provide support for the research on preventing and combating red envelope fraud. By understanding the existing classical methods, looking for solutions and experiments on the existing methods, in order to find more solutions suitable for this task and improve the performance of the model.

### 1.3 关键词 Keyword

**深度学习：**利用深度神经网络结构自动学习和提取图像中的抽象特征，实现红包图像的识别和检测。
**图像分类：**采用深度学习图像识别中的经典模型，处理和分析图像数据，实现红包图像的分类处理。

**Deep learning:** The use of deep neural network structure to automatically learn and extract abstract features in the image to realize the recognition and detection of the red envelope image.
**Image classification:** The classical model of deep learning image recognition is used to process and analyze image data to realize the classification and processing of red envelope images.

### 1.4 项目结构 Project Structure

```Document Map
graduation_project:.
├─.gitignore
├─README.md
└─graduation_project_program 	// 主程序 Main program
    ├─Augmentation 				// 自定义图像增强 Custom image enhancement
    │  └─data_augment.py
    ├─dataset
    │  ├─test
    │  │  └─img					// .jpg
    │  └─train 					// YOLO格式数据集 YOLO format data set
    │      ├─img				// .jpg
    │      └─label				// .txt
    ├─log 						// 日志文件夹 Log folder
    ├─utils 					// 自定义工具函数 Custom utility functions
    │  └─myloss.py	
    ├─weights 					// 模型参数文件夹 Model parameter folder
    ├─train.py 					// 训练主文件 Training master file
	├─inference.py 				// 推理主文件 Inference master file
	└─model_test.py				// 个人学习、调试文件 Personal learning, debugging files
```

### 1.5 使用方式

#### 1.5.1 模型训练

pycharm打开graduation_project_program文件夹，运行train.py训练模型。  
或者在命令行中运行train.py训练模型。

#### 1.5.2 模型测试

pycharm打开graduation_project_program文件夹，运行inference.py测试模型。  
或者在命令行中运行inference.py测试模型。

## 2. 参考资料 References

[1]苏思铭,王浩宇,徐国爱.基于深度学习的网络广告视觉欺诈检测方法[J].  2021.  
[2]郑远攀,李广阳,李晔.深度学习在图像识别中的应用研究综述[J].计算机工程与应用, 2019, 55(12):17.DOI:10.3778/j.issn.1002-8331.1903-0031.  
[3]宁静涛,苏达新.深度学习在图像识别中的应用与挑战探析[J].电脑知识与技术,2023,19(28):24-26.DOI:10.14004/j.cnki.ckt.2023.1503.  
[4]Lecun Y , Bottou L .Gradient-based learning applied to document recognition[J].Proceedings of the IEEE, 1998, 86(11):2278-2324.DOI:10.1109/5.726791.  
[5]Krizhevsky A , Sutskever I , Hinton G .ImageNet Classification with Deep Convolutional Neural Networks[J].Advances in neural information processing systems, 2012, 25(2).DOI:10.1145/3065386.  
[6]Zeiler D M ,Fergus R .Visualizing and Understanding Convolutional Networks.[J].CoRR,2013,abs/1311.2901  
[7]Simonyan K ,Zisserman A .Very Deep Convolutional Networks for Large-Scale Image Recognition.[J].CoRR,2014,abs/1409.1556  
[8]Szegedy C ,0015 L W ,Jia Y , et al.Going Deeper with Convolutions.[J].CoRR,2014,abs/1409.4842  
[9]He K ,Zhang X ,Ren S , et al.Deep Residual Learning for Image Recognition.[J].CoRR,2015,abs/1512.03385  
[10]Huang G , Liu Z , Laurens V D M ,et al.Densely Connected Convolutional Networks[J].IEEE Computer Society, 2016.DOI:10.1109/CVPR.2017.243.  
[11]Hu J , Shen L , Sun G .Squeeze-and-Excitation Networks[C]//2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).IEEE, 2018.DOI:10.1109/CVPR.2018.00745.  


## 鸣谢

感谢学院、指导教师和同学对本项目的支持和指导。

