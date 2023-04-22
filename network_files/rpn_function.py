from typing import List, Optional, Dict, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision

from . import det_utils
from . import boxes as box_ops
from .image_list import ImageList


@torch.jit.unused
def _onnx_get_num_anchors_and_pre_nms_top_n(ob, orig_pre_nms_top_n):
    # 此处严格来说并不是注释，不能乱修改，通过此处的type,可以获取输入变量的类型，先进性类型检查，防止运行过程中出现问题
    # type: (Tensor, int) -> Tuple[int, int]
    from torch.onnx import operators
    num_anchors = operators.shape_as_tensor(ob)[1].unsqueeze(0)
    pre_nms_top_n = torch.min(torch.cat(
        (torch.tensor([orig_pre_nms_top_n], dtype=num_anchors.dtype),
         num_anchors), 0))

    return num_anchors, pre_nms_top_n

# 生成anchor
class AnchorsGenerator(nn.Module):
    __annotations__ = {
        "cell_anchors": Optional[List[torch.Tensor]],
        "_cache": Dict[str, List[torch.Tensor]]
    }

    """
    anchors生成器
    Module that generates anchors for a set of feature maps and
    image sizes.

    The module support computing anchors at multiple sizes and aspect ratios
    per feature map.

    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
    per spatial location for feature map i.

    Arguments:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):
    """

    # sizes对应anchor的 scalar，aspect_ratios对用每个anchor所采用的比例，默认值没用到，暂时不用管
    # 在创建mobile版本的模型的时候自己传入的有size和aspect_ratios。创建resnet版本的是没有传入AnchorsGenerator这些参数，搭建的时候是自动生成的，在faster_rcnn_framework代码中的336行可以查看到
    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        super(AnchorsGenerator, self).__init__()
        # 判断zise的每个元素是否是list或者tuple类型的,由tｒain_mobilenet_v2可以发现anchor——generator里面的size是一个tuple.而tuple中的每个元素又是tuple类型，所以此处条件不满足
        if not isinstance(sizes[0], (list, tuple)):
            # TODO change this
            sizes = tuple((s,) for s in sizes)
        # 同样aspect_ratios是一个tuple.而tuple中的每个元素又是tuple类型，依旧不满足条件
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)
        # 都是只有一个元素，满足条件
        assert len(sizes) == len(aspect_ratios)
        # 注意上面的代码，在train_resnet50_fpn的creat_model里面是没有传入anchor_generator这个参数的，在搭建过程中会按照默认值进行生成(在 faster_rcnn_farework325行代码处)

        # 特征层数,mobile版本的只有一层
        self.sizes = sizes
        # anchor比例
        self.aspect_ratios = aspect_ratios
        # 通过下面的set_cell_anchors生成
        self.cell_anchors = None
        # 在原图上生成的所有的anchor的坐标信息放到_cache里面
        self._cache = {}

    def generate_anchors(self, scales, aspect_ratios, dtype=torch.float32, device=torch.device("cpu")):
        # type: (List[int], List[float], torch.dtype, torch.device) -> Tensor
        """
        compute anchor sizes
        Arguments:
            scales: sqrt(anchor_area)
            aspect_ratios: h/w ratios
            dtype: float32
            device: cpu/gpu
        """
        # 将scales转化为tensor，原来是list
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        # 将aspect_ratios开根号就得到每个anchor高度所对应的乘法因子
        h_ratios = torch.sqrt(aspect_ratios)
        # w_ratios对应宽度乘法因子
        w_ratios = 1.0 / h_ratios

        # [r1, r2, r3]' * [s1, s2, s3]
        # number of elements is len(ratios)*len(scales)
        # w_ratios[:, None]表示在宽度乘法因子后面增加一个维度,为了能够进行矩阵相乘
        # 原来w_ratios和scales　都是一个向量，这里的乘法相当于将每个比例都分别成上尺度，这样就得到每一个anchors的宽度值。通过view展平成一维向量，所以number of elements is len(ratios)*len(scales)
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        # left-top, right-bottom coordinate relative to anchor center(0, 0)
        # 生成的anchors模板都是以（0, 0）为中心的, shape [len(ratios)*len(scales), 4]
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

        return base_anchors.round()  # round 四舍五入返回

    def set_cell_anchors(self, dtype, device):
        # type: (torch.dtype, torch.device) -> None
        # 初始化的时候为空，第一次调用的时候肯定为空
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None
            # suppose that all anchors have the same device
            # which is a valid assumption in the current state of the codebase
            if cell_anchors[0].device == device:
                return

        # 根据提供的sizes和aspect_ratios生成anchors模板
        # anchors模板都是以(0, 0)为中心的anchor
        # 将sizes, aspect_ratios传入generate_anchors去生成anchors模板
        # self.sizes的元素个数就是预测特征层的个数,在每个预测特征层上生成anchor
        cell_anchors = [
            self.generate_anchors(sizes, aspect_ratios, dtype, device)
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
        self.cell_anchors = cell_anchors

    def num_anchors_per_location(self):
        # 计算每个预测特征层上每个滑动窗口的预测目标数
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    def grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        """
        anchors position in grid coordinate axis map into origin image
        计算预测特征图对应原始图像上的所有anchors的坐标
        Args:
            grid_sizes: 预测特征矩阵的height和width
            strides: 预测特征矩阵上一步对应原始图像上的步距
        """
        # 定义一个anchor空列表
        anchors = []
        # 将刚才set_cell_anchors得到的每一个预测特征图上的模板赋值给cell_anchors
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None

        # 遍历每个预测特征层的grid_size，strides和cell_anchors
        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            # size对应的就是每个预测特征层的高度和宽度
            grid_height, grid_width = size
            # stride对应的就是每个预测特征层上的一个cell对一个原图上的高度和宽度的尺度信息
            stride_height, stride_width = stride
            # 获取设备信息
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            # shape: [grid_width] 对应原图上的x坐标(列)
            # torch.arange生成序列，从0开始，grid_width个元素个数，将元素个数成上stride_width
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
            # shape: [grid_height] 对应原图上的y坐标(行)
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height

            # 计算预测特征矩阵上每个点对应原图上的坐标(anchors模板的坐标偏移量)
            # torch.meshgrid函数分别传入行坐标和列坐标，生成网格行坐标矩阵和网格列坐标矩阵
            # shape: [grid_height, grid_width]
            # 通过torch.meshgrid得到每个点对应原图上的坐标
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            # 将得到的shift_y, shift_x进行展平处理
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            # 计算anchors坐标(xmin, ymin, xmax, ymax)在原图上的坐标偏移量
            # shape: [grid_width*grid_height, 4]
            # 在维度１上进行shift_x, shift_y, shift_x, shift_y拼接，前两个对应左上角偏移量，后两个对应右下角偏移量，是一样的。简单理解就是网格中的某个点加上偏移量
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            # 将anchors模板与原图上的坐标偏移量相加得到原图上所有anchors的坐标信息(shape不同时会使用广播机制)
            # base_anchors就是之前生成的模板anchors，这里将刚才计算的anchor模板给挨个的放到原图上
            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            anchors.append(shifts_anchor.reshape(-1, 4))

        return anchors  # List[Tensor(all_num_anchors, 4)]

    def cached_grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        """将计算得到的所有anchors信息进行缓存"""
        # 将grid_sizes和strides都转化成字符形式
        key = str(grid_sizes) + str(strides)
        # self._cache是字典类型，初始化的时候为空
        if key in self._cache:
            return self._cache[key]
        # 通过grid_anchors得到所有预测特征层映射会原图上所生成的anchors
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    # image_list就是ImageList这个类(image_list.py文件中的),保存的batch信息和图像缩放后的尺寸信息。feature_maps对应预测特征层的信息，类型是list,里面元素是tensor，list的个数就是预测特征层的个数
    def forward(self, image_list, feature_maps):
        # type: (ImageList, List[Tensor]) -> List[Tensor]
        # 获取每个预测特征层的尺寸(height, width)——>feature_map.shape[-2:]对应高宽
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])

        # 获取输入图像的height和width
        image_size = image_list.tensors.shape[-2:]

        # 获取变量类型和设备类型
        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        # one step in feature map equate n pixel stride in origin image
        # 计算特征层上的一步等于原始图像上的步长。image_size[0]对应原图高度，g[0]对应预测特征图的高度
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]

        # 根据提供的sizes和aspect_ratios生成anchors模板，mobile版本一个模板有15个anchor
        self.set_cell_anchors(dtype, device)

        # 计算/读取所有anchors的坐标信息（这里的anchors信息是映射到原图上的所有anchors信息，不是anchors模板）
        # 得到的是一个list列表，对应每张预测特征图映射回原图的anchors坐标信息
        # 将得到的anchor模板应用到原图上，通过cached_grid_anchors这个函数生成。grid_sizes对应每个预测特层的高度和宽度信息，strides表示每个cell对应原图上的尺度信息
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)

        anchors = torch.jit.annotate(List[List[torch.Tensor]], [])
        # 遍历一个batch中的每张图像
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            # 遍历每张预测特征图映射回原图的anchors坐标信息
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        # 将每一张图像的所有预测特征层的anchors坐标信息拼接在一起
        # anchors是个list，每个元素为一张图像的所有anchors信息
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        # Clear the cache in case that memory leaks.防止内存泄漏
        self._cache.clear()
        return anchors


class RPNHead(nn.Module):
    """
    add a RPN head with classification and regression
    通过滑动窗口计算预测目标概率与bbox regression参数

    Arguments:
        in_channels: number of channels of the input feature
        num_anchors: number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        # 3x3 滑动窗口,in_channels对用输入特征矩阵的channel
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 计算预测的目标分数（这里的目标只是指前景或者背景）,论文中说的是预测2k个分数，这里只预测了k个，都可以，子不过方式不同
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 计算预测的目标bbox regression参数，每个num_anchors预测4个偏移量
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        # 对上面三个卷积层的初始化
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    # ｘ就是通过backbone输出的预测特征层，mobilenet只有一个预测特征层，resnet50有五个
    def forward(self, x):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        logits = []
        bbox_reg = []
        # 遍历每个预测特征层,mobile版本只用了一个特征层，这里只遍历一次
        for i, feature in enumerate(x):
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        # 返回预测的两个列表
        return logits, bbox_reg


def permute_and_flatten(layer, N, A, C, H, W):
    # type: (Tensor, int, int, int, int, int) -> Tensor
    """
    调整tensor顺序，并进行reshape
    Args:
        layer: 预测特征层上预测的目标概率或bboxes regression参数
        N: batch_size
        A: anchors_num_per_position
        C: classes_num or 4(bbox coordinate)
        H: height
        W: width

    Returns:
        layer: 调整tensor顺序，并reshape后的结果[N, -1, C]
    """
    # view和reshape功能是一样的，先展平所有元素在按照给定shape排列
    # view函数只能用于内存中连续存储的tensor，permute等操作会使tensor在内存中变得不再连续，此时就不能再调用view函数
    # reshape则不需要依赖目标tensor是否在内存中是连续的
    # [batch_size, anchors_num_per_position * (C or 4), height, width]
    # 添加一个维度
    layer = layer.view(N, -1, C, H, W)
    # 调换tensor维度，里面的参数对应原始维度的位置
    layer = layer.permute(0, 3, 4, 1, 2)  # [N, H, W, -1, C]
    # 通过permute方法之后，tensor数据在内存中变的不再连续，这里通过reshape再次进行调整，reshape可以用于数据不连续的，view只能用于连续的
    layer = layer.reshape(N, -1, C)
    return layer


# 调整tensor的格式和shape
def concat_box_prediction_layers(box_cls, box_regression):
    # type: (List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    对box_cla和box_regression两个list中的每个预测特征层的预测信息
    的tensor排列顺序以及shape进行调整 -> [N, -1, C]
    Args:
        box_cls: 每个预测特征层上的预测目标概率
        box_regression: 每个预测特征层上的预测目标bboxes regression参数

    Returns:

    """
    # 创建两个空列表，一个用来存储目标分数的参数，一个用来存储预测的bodingbox的回归参数
    box_cls_flattened = []
    box_regression_flattened = []

    # 遍历每个预测特征层的目标分数和边界框回归参数
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        # [batch_size, anchors_num_per_position * classes_num, height, width]
        # 注意，当计算RPN中的proposal时，classes_num=1,只区分目标和背景。AxC表示anchor*class,，这里class=1只需要预测时前景还是背景，所以这里就直接等于anchor个数
        N, AxC, H, W = box_cls_per_level.shape
        # # [batch_size, anchors_num_per_position * 4, height, width]
        Ax4 = box_regression_per_level.shape[1]
        # anchors_num_per_position
        A = Ax4 // 4
        # classes_num=1
        C = AxC // A

        # [N, -1, C]
        # 通过permute_and_flatten对参数进行展平处理(为什么要进行这一步的调整？答：这样做之后能够方便后面的将预测值跟生成的anchor进行结合以及在对proposals进行过滤的时候更加的方便)
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        # 将得到的变量存储到box_cls_flattened
        box_cls_flattened.append(box_cls_per_level)

        # [N, -1, C]
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)
    # 对box_cls_flattened列表进行拼接，也就是将多个预测特征层的信息进行拼接，mobile版本只有一层，然后再展平处理
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)  # start_dim, end_dim
    # 回归参数，跟上面一样
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


# 根据生成的anchor和边界框回归参数和目标分数来生成proposals
class RegionProposalNetwork(torch.nn.Module):
    """
    Implements Region Proposal Network (RPN).

    Arguments:
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): module that computes the objectness and regression deltas
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        pre_nms_top_n (Dict[str]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        post_nms_top_n (Dict[str]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals

    """
    # 对初始化函数中的使用到的变量进行注释，可有可无，就是解释作用
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
        'pre_nms_top_n': Dict[str, int],
        'post_nms_top_n': Dict[str, int],
    }

    # 这些初始化参数都是在faster_rcnn_framework中传过来的，在faster_rcnn_framework这个脚本中，我们对这部分没介绍，下面我们来逐一分析
    # anchor_generator通过AnchorsGenerator这个类产生
    # head是RPNHead这个类
    # fg_iou_thresh, bg_iou_thresh：采集正负样本的阈值
    # batch_size_per_image, positive_fraction表示RPN在计算损失的时候采用的正负样本的总个数
    # pre_nms_top_n, post_nms_top_n, nms_thresh分别标志NMS处理前和NMS处理后的目标数
    # score_thresh表示NMS处理时所采用的阈值
    def __init__(self, anchor_generator, head,
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 pre_nms_top_n, post_nms_top_n, nms_thresh, score_thresh=0.0):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        # BoxCoder类下面再讲
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # use during training
        # 计算anchors与真实bbox的iou
        self.box_similarity = box_ops.box_iou

        # Matcher这个类后面再介绍
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,  # 当iou大于fg_iou_thresh(0.7)时视为正样本
            bg_iou_thresh,  # 当iou小于bg_iou_thresh(0.3)时视为负样本
            # 设置为true表示启用第二条正样本匹配准则
            allow_low_quality_matches=True
        )

        # 后面用的时候在介绍
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction  # 256, 0.5
        )

        # use during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        # 过滤proposals的时候使用的参数
        self.min_size = 1.

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def assign_targets_to_anchors(self, anchors, targets):
        # type: (List[Tensor], List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor]]
        """
        计算每个anchors最匹配的gt，并划分为正样本，背景以及废弃的样本
        Args：
            anchors: (List[Tensor])
            targets: (List[Dict[Tensor])
        Returns:
            labels: 标记anchors归属类别（1, 0, -1分别对应正样本，背景，废弃的样本）
                    注意，在RPN中只有前景和背景，所有正样本的类别都是1，0代表背景
            matched_gt_boxes：与anchors匹配的gt
        """
        # 创建两个空列表用来存储匹配的标签和gt_box
        labels = []
        matched_gt_boxes = []
        # 遍历每张图像的anchors和targets
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            # 对于每张图像只将每个box的坐标信息提取出来，因为rpn只区分是前景还是背景，不管类别
            gt_boxes = targets_per_image["boxes"]
            # 判断gt_boxes有没有元素，一般训练图片都是标注好，肯定有的，这里不用管，直接进入else语句
            if gt_boxes.numel() == 0:
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                # 计算anchors与真实bbox的iou信息
                # set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                # 计算anchor和gt的iou,会计算每个anchor和gt的交并比，计算结果是个混淆矩阵
                match_quality_matrix = box_ops.box_iou(gt_boxes, anchors_per_image)
                # 计算每个anchors与gt匹配iou最大的索引（如果iou<0.3索引置为-1，0.3<iou<0.7索引为-2）
                # 通过proposal_matcher对刚才计算的iou值做进一步处理，通过刚才计算的iou值来为每一个anchor分配他所匹配到的gt_box
                # matched_idxs里面的数值都是-1，-2和大于等于0的部分，大于等于0的即我们匹配到的正样本
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                # 这里使用clamp设置下限0是为了方便取每个anchors对应的gt_boxes信息
                # 负样本和舍弃的样本都是负值，所以为了防止越界直接置为0
                # 因为后面是通过labels_per_image变量来记录正样本位置的，
                # 所以负样本和舍弃的样本对应的gt_boxes信息并没有什么意义，
                # 反正计算目标边界框回归损失时只会用到正样本。
                # 对matched_idxs设置一个下限作为索引传给gt_boxes得到每个anchor匹配到的对应的box坐标，相当于都设置为索引0的gt_box，没影响，后面我们计算损失的时候只计算正样本的
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                # 记录所有anchors匹配后的标签(正样本处标记为大于等于0，负样本处标记为-1，丢弃样本处标记为-2)，这里得到正样本对应的蒙版
                labels_per_image = matched_idxs >= 0
                # 转为float32类型，转型后正样本就对应给你为1，其他为0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # background (negative examples)
                # matched_idxs=-1的也创建一个蒙版，负样本创建一个蒙版
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                # 背景位置全部赋值为0.0
                labels_per_image[bg_indices] = 0.0

                # discard indices that are between thresholds
                # 找到丢弃的置为-1
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_per_image[inds_to_discard] = -1.0
            # 上面得到的值添加到labels中
            labels.append(labels_per_image)
            # anchor匹配到的gt坐标添加到matched_gt_boxes中
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        # type: (Tensor, List[int]) -> Tensor
        """
        获取每张预测特征图上预测概率排前pre_nms_top_n的anchors索引值
        Args:
            objectness: Tensor(每张图像的预测目标概率信息 )
            num_anchors_per_level: List（每个预测特征层上的预测的anchors个数）
        Returns:

        """
        r = []  # 记录每个预测特征层上预测目标概率前pre_nms_top_n的索引信息
        offset = 0
        # 遍历每个预测特征层上的预测目标概率信息,在第1维度(anchor个数)上分割，num_anchors_per_level存储的是预测特征层的目标个数
        for ob in objectness.split(num_anchors_per_level, 1):
            # 这个不用管，不满足
            if torchvision._is_tracing():
                num_anchors, pre_nms_top_n = _onnx_get_num_anchors_and_pre_nms_top_n(ob, self.pre_nms_top_n())
            else:
                num_anchors = ob.shape[1]  # 预测特征层上的预测的anchors个数
                # 针对每一层取前top n个，判断两个最小值，最小多少取多少
                pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)

            # Returns the k largest elements of the given input tensor along a given dimension
            # 对anchor的前2000个进行排序并返回出索引号
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        # type: (Tensor, Tensor, List[Tuple[int, int]], List[int]) -> Tuple[List[Tensor], List[Tensor]]
        """
        筛除小boxes框，nms处理，根据预测概率获取前post_nms_top_n个目标
        Args:
            proposals: 预测的bbox坐标
            objectness: 预测的目标概率
            image_shapes: batch中每张图片的size信息
            num_anchors_per_level: 每个预测特征层上预测anchors的数目

        Returns:

        """
        # 获取图像个数
        num_images = proposals.shape[0]
        # 获取设备信息
        device = proposals.device

        # do not backprop throught objectness
        # 丢弃梯度信息，只获取数值信息
        objectness = objectness.detach()
        # reshape处理，每个图片的proposals分开放[batch, anchor个数]
        objectness = objectness.reshape(num_images, -1)

        # Returns a tensor of size size filled with fill_value
        # levels负责记录分隔不同预测特征层上的anchors索引信息，用于记录对应的anchor在对应的哪些特征层上面
        # torch.full((n,), idx, dtype=torch.int64, device=device表示用idx来填充这矩阵，也就是0,1,2,3,4
        levels = [torch.full((n,), idx, dtype=torch.int64, device=device)
                  # idx表示预测特征层的索引，n表示该层特征层anchor个数，mobile版本的只有一层特征层，这里就循环一次
                  for idx, n in enumerate(num_anchors_per_level)]
        # 在第0维度上拼接刚才生成的矩阵，为了后面来区分anchors所在的特征层的位置，位于哪个特征层上
        levels = torch.cat(levels, 0)

        # Expand this tensor to the same size as objectness
        # expand_as(objectness)方法将tensor的信息进行复制
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        # 获取每张预测特征图上预测概率排前pre_nms_top_n的anchors索引值
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        # 根据一批数据图像个数产生一个列表
        image_range = torch.arange(num_images, device=device)
        # 加一个维度
        batch_idx = image_range[:, None]  # [batch_size, 1]

        # 根据每个预测特征层预测概率排前pre_nms_top_n的anchors索引值获取相应概率信息
        # 根据切片的方式将排序后的目标信息提取出来
        objectness = objectness[batch_idx, top_n_idx]
        # levels存储的是anchor所对应的某一特征层的预测信息，同样使用切片的方式获取目标索引值
        levels = levels[batch_idx, top_n_idx]
        # 预测概率排前pre_nms_top_n的anchors索引值获取相应bbox坐标信息，同样的方式获取proposals
        proposals = proposals[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)

        # 定义两个空列表用于存储最终的分数和位置信息
        final_boxes = []
        final_scores = []
        # 遍历每张图像的相关预测信息，通过zip方法同时遍历proposals, objectness_prob, levels, image_shapes
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            # 调整预测的boxes信息，将越界的坐标调整到图片边界上，对boxes进行裁剪。将proposals限制在图像的内部
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

            # 返回boxes满足宽，高都大于min_size的索引
            # 通过remove_small_boxes将proposals中的小目标删除
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            # 获取滤除小目标之后的proposals
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # 移除小概率boxes，参考下面这个链接
            # https://github.com/pytorch/vision/pull/3205
            keep = torch.where(torch.ge(scores, self.score_thresh))[0]  # ge: >=
            # 通过切片的方式获取滤出小目标后的ｐｒｏｐｏｓａｌ
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            # NMS处理，boxes, scores, lvl, self.nms_thresh分别表示proposals坐标参数，目标分数，level变量，阈值
            # 得到的keep是执行NMS处理之后并且按照目标类别分数进行排序输出的索引信息
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

            # keep only topk scoring predictions
            # 获取前post_nms_top_n个目标
            keep = keep[: self.post_nms_top_n()]
            # 得到最终的预测分数和目标
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        计算RPN损失，包括类别损失（前景与背景），bbox regression损失
        Arguments:
            objectness (Tensor)：预测的前景概率
            pred_bbox_deltas (Tensor)：预测的bbox regression
            labels (List[Tensor])：真实的标签 1, 0, -1（batch中每一张图片的labels对应List的一个元素中）
            regression_targets (List[Tensor])：真实的bbox regression

        Returns:
            objectness_loss (Tensor) : 类别损失
            box_loss (Tensor)：边界框回归损失
        """
        # 按照给定的batch_size_per_image, positive_fraction选择正负样本，返回的是一个正负样板的位置索引蒙版
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        # 将一个batch中的所有正负样本List(Tensor)分别拼接在一起，并获取非零位置的索引
        # sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        # 将正样本蒙版进行拼接求非零元素的索引
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        # sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)
        # 将负样本蒙版进行拼接求非零元素的索引
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        # 将所有正负样本索引拼接在一起，一个batch的拼接在一起
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        # 将预测的目标分数进行展平
        objectness = objectness.flatten()

        # 将标签按照batch维度拼接
        labels = torch.cat(labels, dim=0)
        # 同理将回归 参数进行拼接
        regression_targets = torch.cat(regression_targets, dim=0)

        # 计算边界框回归损失，只需要计算正样本的损失
        box_loss = det_utils.smooth_l1_loss(
            # 预测的bounding box的回归参数所对应的正样本预测值取出来，即预测的正样本的回归参数
            pred_bbox_deltas[sampled_pos_inds],
            # 将每个anchor所对应的gt_box相对于anchor的回归参数所对应的正样本位置的回归参数取出来，即真实的gt相对于anchor的回归参数
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        # 计算目标预测概率损失，传入的预测值没进行热河处理，不需要经过sigmoid，这个方法会自动进行sigmoid
        objectness_loss = F.binary_cross_entropy_with_logits(
            # 预测的目标分数
            objectness[sampled_inds],
            # 真实目标分数
            labels[sampled_inds]
        )

        return objectness_loss, box_loss

    def forward(self,
                images,  # type: ImageList
                # features是传入的预测特征层的特征矩阵
                features,  # type: Dict[str, Tensor]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Tensor], Dict[str, Tensor]]
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (Dict[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[Tensor]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        # features是所有预测特征层组成的OrderedDict
        # 将预测特征层的特征矩阵都给提取出来。mobile有一层，resnet50有五层，这里提取特征层
        features = list(features.values())

        # 计算每个预测特征层上的预测目标概率和bboxes regression参数
        # objectness和pred_bbox_deltas都是list
        # 将预测特征层通过RPNHead生成目标分数和边界框参数
        objectness, pred_bbox_deltas = self.head(features)

        # 生成一个batch图像的所有anchors信息,list(tensor)元素个数等于batch_size
        anchors = self.anchor_generator(images, features)

        # batch_size。求得一个batch中有多少张图片
        num_images = len(anchors)

        # numel() Returns the total number of elements in the input tensor.
        # 计算每个预测特征层上的对应的anchors数量
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]

        # 调整内部tensor格式以及shape
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness,
                                                                    pred_bbox_deltas)

        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        # 将预测的bbox regression参数应用到anchors上得到proposals的坐标
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        # 将计算的proposalsview处理，每个图片单独拎出来
        proposals = proposals.view(num_images, -1, 4)

        # 筛除小boxes框，nms处理，根据预测概率获取前post_nms_top_n个目标
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            assert targets is not None
            # 计算每个anchors最匹配的gt，并将anchors进行分类，前景，背景以及废弃的anchors
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            # 结合anchors以及对应的gt，计算regression参数
            # 将匹配到的gt坐标与anchors对比来求得gt_box对应的anchor的回归参数
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            # 损失计算
            # objectness, pred_bbox_deltas, labels, regression_targets分别为预测的目标分数；预测的目标偏移量；真实的labels(正样本为1，负样本为0)；每个anchor对应的相对应的gt_box相对于anchor的回归参数
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg
            }
        #     返回损失
        return boxes, losses
