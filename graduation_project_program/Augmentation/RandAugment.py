from torchvision import transforms
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


class RandAugment:
    def __init__(self, n: int = 2, m: int = 5):
        """
        初始化 RandAugment
        :param n: 要应用的图像增强运算的数量
        :param m: 所有操作的强度 [0, 10]
        """
        self.n = n
        self.m = m
        self.augmentations = [
            self._shear_x, self._shear_y, self._translate_x, self._translate_y, self._rotate, self._color,
            self._posterize, self._solarize, self._contrast, self._sharpness, self._brightness, self._autocontrast,
            self._equalize
        ]

    def _shear_x(self, img, magnitude):
        """
        对图像进行水平错切
        :param img: PIL Image
        :param magnitude: 剪切强度
        :return: PIL Image
        """
        if magnitude == 0:
            return img
        shear_angle = magnitude * 0.1  # 计算错切角度
        return img.transform(img.size, Image.AFFINE, (1, shear_angle, 0, 0, 1, 0))

    def _shear_y(self, img, magnitude):
        """
        对图像进行垂直错切
        :param img: PIL Image
        :param magnitude: 剪切强度
        :return: PIL Image
        """
        if magnitude == 0:
            return img
        shear_angle = magnitude * 0.1  # 计算错切角度
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, shear_angle, 1, 0))

    def _translate_x(self, img, magnitude):
        """
        对图像进行水平平移
        :param img: PIL Image
        :param magnitude: 平移强度
        :return: PIL Image
        """
        if magnitude == 0:
            return img
        translate_distance = magnitude * img.size[0] * 0.1  # 计算偏移量
        return img.transform(img.size, Image.AFFINE, (1, 0, translate_distance, 0, 1, 0))

    def _translate_y(self, img, magnitude):
        """
        对图像进行垂直平移
        :param img: PIL Image
        :param magnitude: 平移强度
        :return: PIL Image
        """
        if magnitude == 0:
            return img
        translate_distance = magnitude * img.size[1] * 0.1  # 计算偏移量
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, translate_distance))

    def _rotate(self, img, magnitude):
        """
        对图像进行旋转
        :param img: PIL Image
        :param magnitude: 旋转强度
        :return: PIL Image
        """
        if magnitude == 0:
            return img
        # 计算旋转角度
        rotate_angle = magnitude * 30  # magnitude为[0,10], rotate_angle为[0,300]
        return img.rotate(rotate_angle, resample=Image.BILINEAR)

    def _color(self, img, magnitude):
        """
        对图像进行颜色调整
        :param img: PIL Image
        :param magnitude: 调整强度
        :return: PIL Image
        """
        if magnitude == 0:
            return img
        color_factor = magnitude * 0.5  # 调整饱和度 magnitude 0~10 对应 0~1~3
        return ImageEnhance.Color(img).enhance(color_factor)

    def _posterize(self, img, magnitude):
        """
        对图像进行降低色彩深度
        :param img: PIL Image
        :param magnitude: 降低色彩深度强度
        :return: PIL Image
        """
        if magnitude == 0:
            return img
        # 海报化，减少颜色深度
        bits = int(- magnitude * 0.7 + 8)  # bits（1-8）每个像素保留的位数， magnitude 0~10 对应 bits 8~1
        return ImageOps.posterize(img, bits)

    def _solarize(self, img, magnitude):
        """
        对图像进行光照变化
        :param img: PIL Image
        :param magnitude: 光照变化强度
        :return: PIL Image
        """
        if magnitude == 0:
            return img
        # 曝光过度处理
        solarize_threshold = int(- magnitude * 25.5 + 255)  # magnitude 0~10 对应 solarize_threshold 255~0
        return ImageOps.solarize(img, solarize_threshold)

    def _contrast(self, img, magnitude):
        """
        对图像进行对比度调整
        :param img: PIL Image
        :param magnitude: 调整强度
        :return: PIL Image
        """
        if magnitude == 0:
            return img
        contrast_factor = magnitude * 0.5  # 调整对比度 magnitude 0~10 对应 0~1~5
        return ImageEnhance.Contrast(img).enhance(contrast_factor)

    def _sharpness(self, img, magnitude):
        """
        对图像进行锐度调整
        :param img: PIL Image
        :param magnitude: 调整强度
        :return: PIL Image
        """
        if magnitude == 0:
            return img
        sharpness_factor = magnitude  # 调整锐度 magnitude 0~10 对应 0~1~10
        return ImageEnhance.Sharpness(img).enhance(sharpness_factor)

    def _brightness(self, img, magnitude):
        """
        对图像进行亮度调整
        :param img: PIL Image
        :param magnitude: 调整强度
        :return: PIL Image
        """
        if magnitude == 0:
            return img
        brightness_factor = magnitude * 0.5  # 调整亮度 magnitude 0~10 对应 0~1~3
        return ImageEnhance.Brightness(img).enhance(brightness_factor)

    def _autocontrast(self, img, magnitude):
        """
        对图像进行自动对比度调整
        :param img: PIL Image
        :param magnitude:
        :return: PIL Image
        """
        if magnitude == 0:
            return img
        # 自动调整对比度
        return ImageOps.autocontrast(img)

    def _equalize(self, img, magnitude):
        """
        对图像进行直方图均衡化
        :param img: PIL Image
        :param magnitude:
        :return: PIL Image
        """
        if magnitude == 0:
            return img
        # 直方图均衡化
        return ImageOps.equalize(img)

    def _apply_augmentation(self, img, name):
        """
        以随机强度应用指定图像增强
        :param img: PIL Image
        :param name: 图像增强名称
        :return: PIL Image
        """
        magnitude = float(self.m) * np.random.uniform(0, 1)
        # print("magnitude:", magnitude)
        return name(img, magnitude)

    def __call__(self, img):
        """
        对图像进行随机增强
        :param img: PIL Image
        :return: PIL Image
        """
        # 随机选择图像增强操作(n次)
        for _ in range(self.n):
            augmentation = np.random.choice(self.augmentations)
            # print(augmentation.__name__)
            img = self._apply_augmentation(img, augmentation)
        return img


if __name__ == '__main__':
    # 实例化 RandAugment
    randaugment = RandAugment(n=2, m=9)
    # 定义数据增强
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        randaugment,
        transforms.ToTensor(),
    ])
    # 读取图片
    img = Image.open('test.jpg')
    # 对图片进行随机增强
    augmented_img = randaugment(img)
    # 显示增强后的图片
    augmented_img.show()
    # 保存增强后的图片
    augmented_img.save('augmented_test.jpg')
