from PIL import Image

import torchvision
from torchvision.transforms import AutoAugmentPolicy

from Augmentation.RandAugment import RandAugment


class DataAugment(object):
    def __init__(self, size: tuple = (224, 224), type: str = None, **kwargs):
        """
        DataAugment class
        :param size: 图片尺寸，默认为 (224, 224)
        :param type: 数据增强类型，可选 'autoaugment' 或 'randaugment'，默认为 None
        :param kwargs: 其他参数，如 autoaugment 的参数 policy='imagenet'
        """
        super(DataAugment, self).__init__()
        self.size = size
        self.type = type
        self.kwargs = kwargs
        self.aug = None
        if self.type == "autoaugment":
            if self.kwargs.get("policy", None) is None or self.kwargs.get("policy", None) == "imagenet":
                self.kwargs["policy"] = AutoAugmentPolicy.IMAGENET
            elif self.kwargs.get("policy", None) == "cifar10":
                self.kwargs["policy"] = AutoAugmentPolicy.CIFAR10
            elif self.kwargs.get("policy", None) == "svhn":
                self.kwargs["policy"] = AutoAugmentPolicy.SVHN
            self.aug = torchvision.transforms.AutoAugment(**self.kwargs)
        elif type == "randaugment":
            self.aug = RandAugment(n=2, m=5)

        # 数据增强类型参数的有效性检查
        if self.type not in {None, "autoaugment", "randaugment"}:
            raise ValueError("数据增强类型参数 'type' 必须为 'autoaugment', 'randaugment' 或 None.")

    def detect_resize(self, img, label):
        """
        # 改变图片尺寸
        :param img: 图片
        :param label: 标签
        :return: 图片和标签
        """
        trans = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(size=self.size),
            torchvision.transforms.ToTensor()]
        )
        if self.aug is not None:
            trans = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize(size=self.size),
                self.aug,
                torchvision.transforms.ToTensor()]
            )
        image = trans(img)
        return image, label

    # 重定意可以用()调用; ()==__call__; a.__call__===a()
    def __call__(self, *args, **kwargs):
        '''
        定义函数时，*代表收集参数，**代表收集关键字参数，合成元组或字典
        调用函数时，传参带有*和**用于分配参数，拆分元组或字典
        :param args:  __call__(1,2,3)获得args==(1,2,3)
        :param kwargs:  __call__(x=1,y=2,z=3)获得kwargs=={'z':3,'x':1,'y':2}
        :return:
        '''
        return self.detect_resize(*args)


if __name__ == '__main__':
    img = Image.open('test.jpg')
    label = 1

    trans = torchvision.transforms.ToTensor()

    img = trans(img)

    # data_augment = DataAugment(size=(224, 224))
    # data_augment = DataAugment(size=(224, 224), type="randaugment")
    # data_augment = DataAugment(size=(224, 224), type="autoaugment", policy="imagenet")
    # data_augment = DataAugment(size=(224, 224), type="autoaugment", policy="cifar10")
    data_augment = DataAugment(size=(224, 224), type="autoaugment", policy="svhn")

    img, label = data_augment(img, label)

    trans = torchvision.transforms.ToPILImage()
    img = trans(img)

    img.show()
