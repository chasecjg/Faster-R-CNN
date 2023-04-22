import random
from torchvision.transforms import functional as F


class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    # 传入图像和标签文件
    def __call__(self, image, target):
        # 随机生成一个概率，如果小于0.5就进行翻转
        if random.random() < self.prob:
            # 获取图像高宽
            height, width = image.shape[-2:]
            # 水平翻转
            image = image.flip(-1)  # 水平翻转图片
            bbox = target["boxes"]
            # 翻转标签的位置
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            # 更新标签信息
            target["boxes"] = bbox
        return image, target
