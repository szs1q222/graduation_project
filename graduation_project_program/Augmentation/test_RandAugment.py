from PIL import Image, ImageOps, ImageEnhance


# randaugment模块内函数测试

def test(img, magnitude):
    if magnitude == 0:
        return img
    sharpness_factor = magnitude  # 调整锐度 magnitude 0~10 对应 0~1~10
    return ImageEnhance.Sharpness(img).enhance(sharpness_factor)


# 示例使用
original_image = Image.open('test.jpg')
sheared_image = test(original_image, 1)
sheared_image.show()
sheared_image = test(original_image, 5)
sheared_image.show()
sheared_image = test(original_image, 10)
sheared_image.show()
