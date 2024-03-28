from PIL import Image

import numpy as np
import torchvision


class DataAugment(object):
    def __init__(self, size: tuple = (224, 224)):
        super(DataAugment, self).__init__()
        self.size = size

    def detect_resize(self, img, label):  # 改变图片尺寸
        trans = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(size=self.size),
            torchvision.transforms.ToTensor()]
        )
        image = trans(img)
        return image, label

    # 重定意可以用()调用; ()==__call__; a.__call__===a()
    def __call__(self, *args, **kwargs):  # args:tuple    kwargs:dict
        '''
        定义函数时，*代表收集参数，**代表收集关键字参数，合成元组或字典
        调用函数时，传参带有*和**用于分配参数，拆分元组或字典
        :param args:  __call__(1,2,3)获得args==(1,2,3)
        :param kwargs:  __call__(x=1,y=2,z=3)获得kwargs=={'z':3,'x':1,'y':2}
        :return:
        '''
        return self.detect_resize(*args)
