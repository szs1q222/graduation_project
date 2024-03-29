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
	├─test.py 					// 测试主文件 Test master file
	├─reference.py				// 个人相关参考模板 Personal reference template
	└─model_test.py				// 个人学习、调试文件 Personal learning, debugging files
```

### 1.5 使用方式



## 鸣谢

感谢学院和指导教师对本项目的支持和指导，感谢参与本项目的团队成员的辛勤付出。

