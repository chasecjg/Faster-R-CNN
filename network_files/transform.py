import math
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn, Tensor
import torchvision

from .image_list import ImageList




@torch.jit.unused
def _resize_image_onnx(image, self_min_size, self_max_size):
    # type: (Tensor, float, float) -> Tensor
    from torch.onnx import operators
    im_shape = operators.shape_as_tensor(image)[-2:]
    # 获取长宽中的最小值
    min_size = torch.min(im_shape).to(dtype=torch.float32)
    # 获取长宽中的最大值
    max_size = torch.max(im_shape).to(dtype=torch.float32)
    # 最小缩放因子
    scale_factor = torch.min(self_min_size / min_size, self_max_size / max_size)
    # 对图像进行缩放，输入的就是image，image[none]表示在输入的image的前面再加一个维度，由原来的[C, H ,W]->[1, C, H ,W],因为双线性插值只支持4维运算。运算完之后再转回来。通过切片转，最后的[0]就是赚回来3维
    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor, mode="bilinear", recompute_scale_factor=True,
        align_corners=False)[0]

    return image


def _resize_image(image, self_min_size, self_max_size):
    # type: (Tensor, float, float) -> Tensor
    im_shape = torch.tensor(image.shape[-2:])
    min_size = float(torch.min(im_shape))    # 获取高宽中的最小值
    max_size = float(torch.max(im_shape))    # 获取高宽中的最大值
    scale_factor = self_min_size / min_size  # 根据指定最小边长和图片最小边长计算缩放比例

    # 如果使用该缩放比例计算的图片最大边长大于指定的最大边长
    if max_size * scale_factor > self_max_size:
        scale_factor = self_max_size / max_size  # 将缩放比例设为指定最大边长和图片最大边长之比

    # interpolate利用插值的方法缩放图片
    # image[None]操作是在最前面添加batch维度[C, H, W] -> [1, C, H, W]
    # bilinear只支持4D Tensor
    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor, mode="bilinear", recompute_scale_factor=True,
        align_corners=False)[0]

    return image

# 对传入的图像进行标准化处理和resize处理，打包成一个个的batch送入网络进行正向传播
class GeneralizedRCNNTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model.

    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """
    # min_size, max_size分别为指定输入网络图像的最小边长和最大边长。image_mean, image_std分别为标准化处理中的均值和方差，对于每张输入的额图像必须缩放到最小值和最大值之间
    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        # 判断min_size是否是list或者是tuplele类型，如果不是就转为tuple类型
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size      # 指定图像的最小边长范围
        self.max_size = max_size      # 指定图像的最大边长范围
        self.image_mean = image_mean  # 指定图像在标准化处理中的均值
        self.image_std = image_std    # 指定图像在标准化处理中的方差

    # 标准化处理函数
    def normalize(self, image):
        """标准化处理"""
        # 获取数据类型和设备信息(cpu还是gpu)
        dtype, device = image.dtype, image.device
        # 将self.image_mean和self.image_std转化成tensor格式
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        # mean传进来的是一个list,list里面有3个元素(image_mean = [0.485, 0.456, 0.406])，上面的语句将其转为tensor之后呢就是shape：[3]的形式，通过添加none将其转为三维tensor,image本来就是三维的(CHW),将mean扩展后就与image的维度一致，在进行标准化处理
        # [:, None, None]: shape [3] -> [3, 1, 1]
        return (image - mean[:, None, None]) / std[:, None, None]

    def torch_choice(self, k):
        # type: (List[int]) -> int
        """
        Implements `random.choice` via torch ops so it can be compiled with
        TorchScript. Remove if https://github.com/pytorch/pytorch/issues/25803
        is fixed.
        """
        index = int(torch.empty(1).uniform_(0., float(len(k))).item())
        return k[index]

    # 将图像限制在所给定的最小值和最大值之间
    def resize(self, image, target):
        # type: (Tensor, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
        """
        将图片缩放到指定的大小范围内，上面有给定最小值和最大值，并对应缩放bboxes信息
        Args:
            image: 输入的图片
            target: 输入图片的相关信息（包括bboxes信息）

        Returns:
            image: 缩放后的图片
            target: 缩放bboxes后的图片相关信息
        """
        # image shape is [channel, height, width]
        # 获取图像的高宽
        h, w = image.shape[-2:]
        # 是判断是训练模式还是验证模式
        if self.training:
            # 从self.min_size中随机选取一个值。其实训练过程中min_size传递的就是一个int类型的，并不是一个列表，只不过强制转换成了元组。其实这里就是将指定的输入网络的最小值赋值给size
            size = float(self.torch_choice(self.min_size))  # 指定输入图片的最小边长,注意是self.min_size不是min_size
        else:
            # FIXME assume for now that testing uses the largest scale
            # 取self.min_size的最后一个元素
            size = float(self.min_size[-1])    # 指定输入图片的最小边长,注意是self.min_size不是min_size

        # 不满足，直接跳过，为了转为onnx格式
        if torchvision._is_tracing():
            image = _resize_image_onnx(image, size, float(self.max_size))
        else:
            # 缩放图像
            image = _resize_image(image, size, float(self.max_size))
        # 判断target是否为空，为空对应的就是验证模式，直接输出图像和target就行了，否者跳到执行下面的语句
        if target is None:
            return image, target
        # 获取边界框参数
        bbox = target["boxes"]
        # 根据图像的缩放比例来缩放bbox，[h, w]表示缩放前的图像的高宽，image.shape[-2:]表示缩放后的尺寸。通过resize_boxes函数之后就将bbox的信息进行了缩放得到新的bbox
        bbox = resize_boxes(bbox, [h, w], image.shape[-2:])
        # 将新的bbox赋值给target["boxes"]所对应的信息
        target["boxes"] = bbox

        return image, target

    # _onnx_batch_images() is an implementation of
    # batch_images() that is supported by ONNX tracing.
    @torch.jit.unused
    def _onnx_batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        max_size = []
        for i in range(images[0].dim()):
            max_size_i = torch.max(torch.stack([img.shape[i] for img in images]).to(torch.float32)).to(torch.int64)
            max_size.append(max_size_i)
        stride = size_divisible
        max_size[1] = (torch.ceil((max_size[1].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size[2] = (torch.ceil((max_size[2].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size = tuple(max_size)

        # work around for
        # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        # which is not yet supported in onnx
        padded_imgs = []
        for img in images:
            padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
            padded_img = torch.nn.functional.pad(img, [0, padding[2], 0, padding[1], 0, padding[0]])
            padded_imgs.append(padded_img)

        return torch.stack(padded_imgs)

    # 返回的maxes存储的就是输入的一个batch的最大高度，最大宽度，最大channel
    def max_by_axis(self, the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    # 将图像打包成一个batch输入到网络中。经过上面的标准化处理和resize之后图像并不是一个统一的大小。为了加速训练，需要将多张图像缩放到统一的尺寸打包成一个tensor送到网络。次出并不是一个简单粗暴的热resize,当然也可以直接resize,只不过这样做不好(不能保持原始图像的比例)
    # 此处的方法是以最大的图像为标准，其他小的图像周围补0
    def batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        """
        将一批图像打包成一个batch返回（注意batch中每个tensor的shape是相同的）
        Args:
            images: 输入的一批图片
            size_divisible: 将图像高和宽调整到该数的整数倍，向上取整，猜测这样做的好处应该就是对硬件比较友好，加速运算

        Returns:
            batched_imgs: 打包成一个batch后的tensor数据
        """
        # 训练的时候此条件不满足，直接跳过。这样做的目的视为了将模型转为onnx模型，onnx是一个开放的神经网络交换格式，可以将各种深度学习框架转为onnx,通过这个格式也可以在各个框架下进行转换
        # 转换为onnx之后就不在依赖于当前的深度学习框架。
        if torchvision._is_tracing():
            # batch_images() does not export well to ONNX
            # call _onnx_batch_images() instead
            return self._onnx_batch_images(images, size_divisible)

        # 分别计算一个batch中所有图片中的最大channel(都是一样的), height, width，遍历每张图片，将他们的shape转为list放到列表中，然后通过max_by_axis找出这一批图像的最大高宽，返回的是max_size是(CHW)
        max_size = self.max_by_axis([list(img.shape) for img in images])

        # 将缩放后的图像都向上取整到32的整数倍,方便填充后的图像进行下采样，因为网络最后要下采样32倍
        stride = float(size_divisible)
        # max_size = list(max_size)
        # 将height向上调整到stride的整数倍
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        # 将width向上调整到stride的整数倍
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        # [batch, channel, height, width]
        # 给max_size(C,H,W)加个维度，batch的个数，就是传入图片的个数
        batch_shape = [len(images)] + max_size

        # 利用new_full方法创建shape为batch_shape且值全部为0的tensor。images[0]取那个值都可以，他就是一个tensor，利用tensor下面的一个方法new_full来创建一个新的tensor,他的shape就是刚才创建的batch_shape，用0填充
        batched_imgs = images[0].new_full(batch_shape, 0)
        # 通过zip方法遍历images和刚才创建的batched_imgs，将传入的每张图片赋值到pad_img，也就是对用batched_imgs。
        for img, pad_img in zip(images, batched_imgs):
            # 将输入images中的每张图片复制到新的batched_imgs的每张图片中，对齐左上角，保证bboxes的坐标不变
            # 这样保证输入到网络中一个batch的每张图片的shape相同
            # copy_: Copies the elements from src into self tensor and returns self
            # 分别对应所有通道，传入图像的高，宽，赋值给pad_img，便利玩之后九江所有图片都赋值给了新创建的空tensor,在上面覆盖传入的图像
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        # 返回填充后的一批图像数据
        return batched_imgs

    # 对网络的预测结果进行后处理（主要将bboxes还原到原图像尺度上）
    # image_shape是将图像resize之后的每一个图像的高度和宽度，original_image_sizes对应每张图像缩放前的尺寸
    # 对应网络的最后一层的generalize
    def postprocess(self,
                    # 网络的最终预测结果
                    result,                # type: List[Dict[str, Tensor]]
                    # resize之后的图像的尺寸
                    image_shapes,          # type: List[Tuple[int, int]]
                    # 缩放前的图像原始尺寸
                    original_image_sizes   # type: List[Tuple[int, int]]
                    ):
        # type: (...) -> List[Dict[str, Tensor]]
        """
        对网络的预测结果进行后处理（主要将bboxes还原到原图像尺度上）
        Args:
            result: list(dict), 网络的预测结果, len(result) == batch_size
            image_shapes: list(torch.Size), 图像预处理缩放后的尺寸, len(image_shapes) == batch_size
            original_image_sizes: list(torch.Size), 图像的原始尺寸, len(original_image_sizes) == batch_size

        Returns:

        """
        # 训练模式直接跳过，不用后处理，训练只需要损失，反向传播就行了
        if self.training:
            return result

        # 遍历每张图片的预测信息，将boxes信息还原回原尺度
        # pred对应每张图像的预测信息, im_s表示缩放后的尺寸, o_im_s缩放前的尺寸
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            # 获取预测的boxes的信息
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)  # 将bboxes缩放回原图像尺度上
            result[i]["boxes"] = boxes
        return result

    def __repr__(self):
        """自定义输出实例化对象的信息，可通过print打印实例信息"""
        format_string = self.__class__.__name__ + '('
        _indent = '\n    '
        format_string += "{0}Normalize(mean={1}, std={2})".format(_indent, self.image_mean, self.image_std)
        format_string += "{0}Resize(min_size={1}, max_size={2}, mode='bilinear')".format(_indent, self.min_size,
                                                                                         self.max_size)
        format_string += '\n)'
        return format_string

    def forward(self,
                images,       # type: List[Tensor]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]
        #遍历每张图片得到一个列表
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            # 如果target不为空就将 targets[i]赋值给 target_index，为空的话就等于none
            target_index = targets[i] if targets is not None else None

            # 判断图像是否是rgb彩色图像
            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            image = self.normalize(image)                # 对图像进行标准化处理
            # 对图像和对应的bboxes缩放到指定范围，图像缩放之后对应的label也要进行重新修正
            image, target_index = self.resize(image, target_index)
            # 将resize之后的图像重新赋值给image中索引为i的进行替换
            images[i] = image
            # 判断targrt是否为空，不为空的话就将target_index赋值给targets中索引为i的targrt进行替换
            if targets is not None and target_index is not None:
                targets[i] = target_index

        # 记录resize后的图像尺寸
        # 注意此处的image还没进行打包处理，每张图像的尺寸大小还不一样
        image_sizes = [img.shape[-2:] for img in images]
        # 将resize之后的images打包成一个batch(tensor)，注意传入的参数是一个list，里面存放的是image tensor
        images = self.batch_images(images)
        # 此处的定义似乎有点多此一举
        image_sizes_list = torch.jit.annotate(List[Tuple[int, int]], [])

        # 对image_size检查是否有问题并打包成tuple放到image_sizes_list中
        for image_size in image_sizes:
            assert len(image_size) == 2
            # 打包成一个tuple放到image_sizes_list
            image_sizes_list.append((image_size[0], image_size[1]))

        # images打包之后一个独立的tensor，image_sizes_list是打包之前的高度和宽度信息。为什么要这样做呢？是因为网络预测的边界框信息是resize之后的，二我们要显示的边界框是在原图上的,所以要记录resize之后的信息，原图信息一直知道的
        # ImageList就是存储了图片打包的一个个batch，每个batch是一个tensor
        # images是打包好的tensor。image_sizes_list是打包前的resize之后的图像尺寸。传入的两个参数都是列表
        image_list = ImageList(images, image_sizes_list)
        # 返回image_list和target，就是要输入到backbone的数据
        return image_list, targets

# 传入三个参数，boxes信息，原始图像尺寸，缩放后的图像尺寸
def resize_boxes(boxes, original_size, new_size):
    # type: (Tensor, List[int], List[int]) -> Tensor
    """
    将boxes参数根据图像的缩放情况进行相应缩放

    Arguments:
        original_size: 图像缩放前的尺寸
        new_size: 图像缩放后的尺寸
    """
    # 分别获取在高度方向和宽度方向的缩放因子。通过for循环和zip方法遍历缩放后和缩放前的尺寸分别赋值给s,和s_org
    ratios = [
        # “/”是除法，不是换行
        # 新的尺寸除上旧的尺寸就得到缩放因子，所以ratios就是对应缩放前后高度方向和宽度方向的缩放因子，是个列表，因为new_size, original_size分别对应图像的是高度和宽度
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratios_height, ratios_width = ratios
    # Removes a tensor dimension, boxes [minibatch, 4]
    # Returns a tuple of all slices along a given dimension, already without it.
    # 将边界框的坐标通过unbind()在索引为1的维度上展开， boxes的信息是[minibatch, 4]，有两个维度，第一个维度是minibatch，表示当前图片有几个boxes信息，第二个表示坐标信息：xmin, ymin, xmax, ymax
    # x对应是宽度方向 ，y对应的是宽度 ，进行了boxes坐标的缩放
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratios_width
    xmax = xmax * ratios_width
    ymin = ymin * ratios_height
    ymax = ymax * ratios_height
    # 通过torch.stack方法在维度1上进行合并，又变成了 boxes [minibatch, 4]的shaoe形状
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)








