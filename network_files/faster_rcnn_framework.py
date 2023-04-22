import warnings
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign

from .roi_head import RoIHeads
from .transform import GeneralizedRCNNTransform
from .rpn_function import AnchorsGenerator, RPNHead, RegionProposalNetwork

'''
此模块定义了faster-rcnn的模型框架
'''


# 整体网络结构
class FasterRCNNBase(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    # 传入网络的三个模块部分以及数据变换方式
    def __init__(self, backbone, rpn, roi_heads, transform):
        super(FasterRCNNBase, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    # 传入两个参数，需要预测的图片，根据下面的type可以知道images是一个list类型，list里面放的是tensor,targets就是对每个图像所标注的信息。这两个采参数就是Dataloader的返回结果
    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        # 判断是否训练模式和target是否为空,训练模式必须使用target
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        # 对传入的数据集进行检查判断是否有错
        # 训练模式，保证targets不为空，否者报错
        if self.training:
            assert targets is not None
            # 对每个targets进行检查
            for target in targets:  # 进一步判断传入的target的boxes参数是否符合规定
                # 获取边界狂参数
                boxes = target["boxes"]
                # 判断边界信息是否是tensor格式
                if isinstance(boxes, torch.Tensor):
                    # 判断边界框的shape是否等于2，不等于就报错，shape是(N, 4)
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        # N表示所对应的图像中有多少个目标
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                            boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        # 定义一个列表original_image_sizes，用来存储每个图像的原始尺寸，torch.jit.annotate(List[Tuple[int, int]], [])用声明original_image_sizes是list类型
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        # 在自定义的dataset里面img已经被转为tensor格式，在pytorch里面的维度排列顺序是BCHW
        # 遍历图像
        for img in images:
            # 获取图像的高和宽，自定义数据读取的时候image已经被转为了tensor（B,C,H,W）
            val = img.shape[-2:]
            # 进一步判断长度是否等于2，防止输入的是个一维向量
            assert len(val) == 2
            # 记录原始图像的尺寸
            original_image_sizes.append((val[0], val[1]))
        #     上面几行代码是为了记录宽高，为了下面进行transforme变化后方便再按照原尺寸给恢复回去
        # original_image_sizes = [img.shape[-2:] for img in images]

        # 此处的images和targets才是真正的batch，经过transform之后才变成真正的batch。之前都是一张张尺寸大小不一样的图片就没法打包成一个个的batch进行并行训练。此处的transform对输入的图像
        # 进行resize之后都将他们统一一个给定大小的tensor里面,后面会详细介绍
        images, targets = self.transform(images, targets)  # 对图像进行预处理

        # print(images.tensors.shape)，经过上面的transform对图像进行预处理之后图像的大小才统一，才能一批数据送入网络
        features = self.backbone(images.tensors)  # 将图像输入backbone得到特征图
        # 判断通过backbone之后特征图是否是tensor类型，mobile—v2只有一个特征图
        if isinstance(features, torch.Tensor):  # 若只在一层特征层上预测，将feature放入有序字典中，并编号为‘0’
            # 将特征图放入有序字典中，此处的输出特征图只有一层，如果是resnet的话则有五个输出特征层，可以看到每层的输出结果
            features = OrderedDict([('0', features)])  # 若在多层特征层上预测，传入的就是一个有序字典

        # 将特征层以及标注target信息传入rpn中
        # proposals: List[Tensor], Tensor_shape: [num_proposals, 4],
        # 每个proposals是绝对坐标，且为(x1, y1, x2, y2)格式
        # 将输入的图像，backbone的输出特征层和标注信息targets传入targets,得到区域建议框和区域建议框损失
        # 生成的proposal是一个列表，他的shape是[num_proposals, 4]，级proposal的个数和，４表示坐标(x1, y1, x2, y2)
        proposals, proposal_losses = self.rpn(images, features, targets)

        # 将rpn生成的数据以及标注target信息传入fast rcnn后半部分即roi-head，得到一系列检测到的目标detections和faster_rcnn的损失值
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        # 对网络的预测结果进行后处理（主要将bboxes还原到原图像尺度上）。images.image_sizes是预处理之后所得到的一系列图像的尺寸，original_image_sizes是预处理前的图像尺寸，将预测结果应映射回原图像尺寸
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        # 损失统计，统计faster-rcnn的损失和rpn损失
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # torch.jit的用法参考：https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html
        # 检查是否在TorchScript模式(此模式不依赖与Python环境)，是的话就返回损失值和检测结果，不是的话调用另一个函数
        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)

        # if self.training:
        #     return losses
        #
        # return detections


# 分数和位置信息预测，就是两个全连接层，进行proposals的选取
class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


# 主要是在初始函数中定义一系列的参数以及上面FastRCNNBase中提到的一系列模块
class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


# FasterRCNN模型实现
class FasterRCNN(FasterRCNNBase):
    """
    Implements Faster R-CNN.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values
          between 0 and H and 0 and W
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values between
          0 and H and 0 and W
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        rpn_score_thresh (float): during inference, only return proposals with a classification score
            greater than rpn_score_thresh
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes

    """

    # 初始化传入参数主要分为三个部分，backbone部分，RPN部分，BOX部分，backbone骨干网络，num_classes是检测目标类别个数，需要加上背景
    def __init__(self, backbone, num_classes=None,
                 # transform parameter
                 min_size=800, max_size=1333,  # 预处理resize时限制的最小尺寸与最大尺寸，就是将输入的图像都限制在这个尺寸范围之内
                 image_mean=None, image_std=None,  # 预处理normalize时使用的均值和方差

                 # RPN parameters
                 # rpn_anchor_generator就是用于生成anchor的生成器，rpn_head对应的是3*3的卷积层和一个分类层，一个边界框回归层，其中3*3的卷积其实就是滑动窗口
                 rpn_anchor_generator=None, rpn_head=None,
                 # 非极大值抑制之前保留的建议框数目，根据预测的分数进行保留的，训练模式下保留2000个，测试模式下保留1000个
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,  # rpn中在nms处理前保留的proposal数(根据score)
                 # 经过非极大致抑制之后所保留的建议框，训练模式下保留2000个，测试模式下保留1000个。这里有点疑惑，我的理解这里是在第二阶段保留的最大候选框数目
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,  # rpn中在nms处理后保留的proposal数
                 rpn_nms_thresh=0.7,  # rpn中进行nms处理时使用的iou阈值
                 # 大于0.7标记为正样本，小于0.3标记为负样本。fg表示前景目标，bg表示背景目镜目标
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,  # rpn计算损失时，采集正负样本设置的阈值
                 # 在正负样本中进行随机采样，总共样256个，正负样本按照1:1
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,  # rpn计算损失时采样的样本数，以及正样本占总样本的比例
                 rpn_score_thresh=0.0,

                 # Box parameters，也就是Roi head部分参数
                 # box_roi_pool，box_head，box_predictor分别对应不同的层
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 # box_score_thresh表示滤出小概率的目标的阈值，小于0.5的话就记为负样本
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 # 分别对应移除低目标概率      fast rcnn中进行nms处理的阈值   对预测结果根据score排序取前100个目标
                 # 在计算faster-rcnn误差所设置的正负阈值，建议狂与gt框大于0.5就记为正样本，
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,  # fast rcnn计算误差时，采集正负样本设置的阈值
                 # 总共采样512个样本，正样本站总样本的0.25，这里是针对第二阶段的损失计算，区别于上面的rpn的采样，rpn是第一阶段的
                 box_batch_size_per_image=512, box_positive_fraction=0.25,  # fast rcnn计算误差时采样的样本数，以及正样本占所有样本的比例
                 bbox_reg_weights=None):
        # 判断backbone有没有out_channels这个属性，out_channel就是对应一个输出特征矩阵的深度，在创建模型的时候是有的。
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels"
                "specifying the number of output channels  (assumed to be the"
                "same for all the levels"
            )
        # 判断定义的rpn_anchor_generator是不是AnchorsGenerator这个类，不传为None也可以下面会创建，这个在创建模型的时候也是有的
        assert isinstance(rpn_anchor_generator, (AnchorsGenerator, type(None)))
        # box_roi_pool对用的就是train_mobilev2里面的roil_pooler,就是ROIpooling层,直接使用的torch.ops里面的方法。所以这里传入的要么为给定的类型要么为空下面自己定义，模型创建的时候默认传入了
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))
        # 自定义了num_class之后就需要自定义box_predictor,此处传入的为不为空，就要进行判断，box_predictor传入的为none，下面会定义
        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor "
                                 "is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        # 预测特征层的channels
        out_channels = backbone.out_channels
        # 在train_mobilev2里面已经初始化过anchor_generator了,此处跳过,但是在train_resnet50的实例化模型(create_model)里面并没有初始化，此处需要执行
        # 若anchor生成器为空，则自动生成针对resnet50_fpn的anchor生成器
        # 此处与mobilev2(train_mobilenetv2里面的create_model的anchor_generator)定义的方式有所不同，他这里面每个地方都是一个元组
        # mobile版本传入的有，这里不用管
        if rpn_anchor_generator is None:
            # resnet50总共5个人预测层，所以这里定义了5个元组(注意逗号不能少)，每个预测特征层预测三种比例(区别于mobile直接一个特征层预测5*3)
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            # * len(anchor_sizes)表示把元素重复5遍
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorsGenerator(
                anchor_sizes, aspect_ratios
            )

        # 生成RPN通过滑动窗口预测网络部分,rpn_head一般为None,rpn_head就是一个3*3的卷积层和两个1*1的卷积层
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        # 默认rpn_pre_nms_top_n_train = 2000, rpn_pre_nms_top_n_test = 1000,
        # 默认rpn_post_nms_top_n_train = 2000, rpn_post_nms_top_n_test = 1000,
        # 放入字典中
        # 表示nms处理之前针对每个预测特征层所保留的目标个数
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        # 表示nms处理之后针对每个预测特征层所剩余的目标个数
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        # 定义整个RPN框架,这里先不介绍，后面介绍RPN的时候在细讲
        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)

        #  Multi-scale RoIAlign pooling
        # box_roi_pool对于mobilev2而言是自己给定的，只在一个特征层上预测。但是对于resnet50而言有5个特征层，不传入此参数，所以需要进行初始化
        if box_roi_pool is None:  # train_resnet50不传入此参数
            # 对应的就是图中的ROIpoolig。此处的MultiScaleRoIAlign和之前讲的ROIpooling有所不同,MultiScaleRoIAlign相比与
            box_roi_pool = MultiScaleRoIAlign(
                # 在哪些特征层进行roi pooling，resnet50有5个特征层的输出
                featmap_names=['0', '1', '2', '3'],
                output_size=[7, 7],
                sampling_ratio=2)

        # fast RCNN中roi pooling后的展平处理两个全连接层部分
        if box_head is None:
            resolution = box_roi_pool.output_size[0]  # 默认等于7
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size
            )

        # 在box_head的输出上预测部分，分别用于预测目标概率和编辑框回归参数
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        # 将roi pooling, box_head以及box_predictor结合在一起，roi-head的定义
        roi_heads = RoIHeads(
            # box
            # box_roi_pool就是ROIPoolig层(实际中使用的是ROIAlign)，官方封装好的
            # box_head就展平和全连接操作
            # box_predictor就是网络中的目标类别分数预测和边界框位置预测，也是两个全连接层
            box_roi_pool, box_head, box_predictor,
            # 正负样本阈值
            box_fg_iou_thresh, box_bg_iou_thresh,  # 0.5  0.5
            # 每张图像选取多少个proposals用来计算损失和正样本所占比例
            box_batch_size_per_image, box_positive_fraction,  # 512  0.25
            # 超参数
            bbox_reg_weights,
            # 后处理的时候使用到的阈值
            box_score_thresh, box_nms_thresh, box_detections_per_img)  # 0.05  0.5  100

        # 图像预处理的均值和方差
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        # 对数据进行标准化，缩放，打包成batch等处理部分
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)
