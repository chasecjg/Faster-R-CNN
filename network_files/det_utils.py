import torch
import math
from typing import List, Tuple
from torch import Tensor


class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        # type: (int, float) -> None
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentage of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        # 遍历每张图像的matched_idxs，也就是刚才传过来的labels
        for matched_idxs_per_image in matched_idxs:
            # >= 1的为正样本, nonzero返回非零元素索引
            # positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            # 找正样本
            positive = torch.where(torch.ge(matched_idxs_per_image, 1))[0]
            # = 0的为负样本
            # negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)
            # 找所有的负样本
            negative = torch.where(torch.eq(matched_idxs_per_image, 0))[0]

            # 指定正样本的数量，总样本取得是256，正样本占0.5，这里也就是128
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            # 如果正样本数量不够就直接采用所有正样本，正样本如果不够128，有多少就取多少
            num_pos = min(positive.numel(), num_pos)
            # 指定负样本数量，总样本-正样本的个数得到负样本的个数
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            # 如果负样本数量不够就直接采用所有负样本，计算真正用于计算的负样本个数，匹配到的负样本和用于计算的负样本的最小值。
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            # Returns a random permutation of integers from 0 to n - 1.
            # 随机选择指定数量的正负样本，将正样本进行随机排序，只取前num_pos个样本，返回的为索引信息
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            # 同上
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            # 根据索引取出正负样本
            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            # 创建一个跟matched_idxs_per_image一样的元素全为0的蒙版用于存储正样本
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            # 创建一个跟matched_idxs_per_image一样的元素全为0的蒙版用于存储负样本
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )

            # 正样本索引位置置1
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            # 负样本索引位置置1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            # 将上面的蒙版添加到列表中
            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx


@torch.jit._script_if_tracing
def encode_boxes(reference_boxes, proposals, weights):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Encode a set of proposals with respect to some
    reference boxes

    Arguments:
        reference_boxes (Tensor): reference boxes(gt)
        proposals (Tensor): boxes to be encoded(anchors)
        weights:
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    # unsqueeze()
    # Returns a new tensor with a dimension of size one inserted at the specified position.
    # proposals为anchor的坐标，左上角和右下角
    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    # 提取每个anchor所对应的gt_box的左上角坐标和右下角坐标
    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # implementation starts here
    # parse widths and heights
    # 计算anchor的宽高
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    # parse coordinate of center point
    # 计算anchor的中心坐标
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    # 同理计算anchor所对应的gt_box的高宽和中心坐标
    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    # 计算ground truth 相对于anchor的回归参数
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    # 拼接起来返回
    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        # type: (Tuple[float, float, float, float], float) -> None
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    # reference_boxes表示每个anchor所对应的gt坐标，proposals为每个anchor的坐标
    def encode(self, reference_boxes, proposals):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        """
        结合anchors和与之对应的gt计算regression参数
        Args:
            reference_boxes: List[Tensor] 每个proposal/anchor对应的gt_boxes
            proposals: List[Tensor] anchors/proposals

        Returns: regression parameters

        """
        # 统计每张图像的anchors个数，方便后面拼接在一起处理后在分开
        # reference_boxes和proposal数据结构相同
        boxes_per_image = [len(b) for b in reference_boxes]
        # 将每张图像对应的gt_box拼接在一起
        reference_boxes = torch.cat(reference_boxes, dim=0)
        # 将anchor也进行拼接
        proposals = torch.cat(proposals, dim=0)

        # targets_dx, targets_dy, targets_dw, targets_dh
        # 传入拼接好的proposals和每个anchor岁对应的gt_box
        targets = self.encode_single(reference_boxes, proposals)
        # 由于上一步计算的参数是拼接之后的，这里要将其分离，分割的长度按照每张图像的anchor个数进行分离，这样有回到各自的图像的参数了
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        # 将权重参数转为tensor，设置的为1，不用管
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)

        # 返回gt_box相对于anchor中心点坐标x,y和高宽的回归参数
        return targets

    # rel_codes预测的目标边界框回归参数，boxes对应anchor坐标
    def decode(self, rel_codes, boxes):
        # type: (Tensor, List[Tensor]) -> Tensor
        """

        Args:
            rel_codes: bbox regression parameters
            boxes: anchors/proposals

        Returns:

        """
        # 判断boxes和rel_codes是不是所指定的类型，不是就报错
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)
        # 获取每张图片的box数目，每批图像可能大小不一样，因为读的数据是随机的
        boxes_per_image = [b.size(0) for b in boxes]
        # 对boxes信息进行拼接，boxes是每张图像岁对应的anchor信息，按照维度0进行拼接，也就是把一个batch的所有anchor坐标信息拼接到一起
        concat_boxes = torch.cat(boxes, dim=0)

        # 求这一个batch的anchor总数，感觉有点多此一举，明明可以直接得到
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val

        # 将预测的bbox回归参数应用到对应anchors上得到预测bbox的坐标
        pred_boxes = self.decode_single(
            rel_codes, concat_boxes
        )

        # 防止pred_boxes为空时导致reshape报错
        if box_sum > 0:
            # 将计算后的proposals重新reshape处理
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)

        return pred_boxes

    def decode_single(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes (bbox regression parameters)
            boxes (Tensor): reference boxes (anchors/proposals)
        """
        # 将boxes这个变量设置成语rel_codes相同的类型
        boxes = boxes.to(rel_codes.dtype)

        # 求得anchor的宽高，中心点坐标，原来anchor的信息是左上角和右下角
        # xmin, ymin, xmax, ymax
        widths = boxes[:, 2] - boxes[:, 0]   # anchor/proposal宽度
        heights = boxes[:, 3] - boxes[:, 1]  # anchor/proposal高度
        ctr_x = boxes[:, 0] + 0.5 * widths   # anchor/proposal中心x坐标
        ctr_y = boxes[:, 1] + 0.5 * heights  # anchor/proposal中心y坐标

        # wx, wy, ww, wh为平衡系数，这里都是1，不用管。SSD目标检测中用了这个
        wx, wy, ww, wh = self.weights  # RPN中为[1,1,1,1], fastrcnn中为[10,10,5,5]
        # 0::4是为了保证采样后dx，dy，dw，dh有两个维度，如果直接使用4的话采样后就一个维度
        dx = rel_codes[:, 0::4] / wx   # 预测anchors/proposals的中心坐标x回归参数
        dy = rel_codes[:, 1::4] / wy   # 预测anchors/proposals的中心坐标y回归参数
        dw = rel_codes[:, 2::4] / ww   # 预测anchors/proposals的宽度回归参数
        dh = rel_codes[:, 3::4] / wh   # 预测anchors/proposals的高度回归参数

        # limit max value, prevent sending too large values into torch.exp()
        # self.bbox_xform_clip=math.log(1000. / 16)   4.135
        # 限制数值的上下限
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        # 将预测值应用到anchor中，根据公式可以直接计算出，这里有个none是因为dx，dy，dw，dh是两个维度
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # 将上面得到的预测结果再转为左上角和右下角的形式
        # xmin
        pred_boxes1 = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymin
        pred_boxes2 = pred_ctr_y - torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        # xmax
        pred_boxes3 = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymax
        pred_boxes4 = pred_ctr_y + torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h

        # 将信息拼接一起，感觉他这里做的很费事(拼接完再展平)，直接concat就可以了
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
        return pred_boxes


class Matcher(object):
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        'BELOW_LOW_THRESHOLD': int,
        'BETWEEN_THRESHOLDS': int,
    }

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        # type: (float, float, bool) -> None
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        # 初始化部分
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold  # 0.7
        self.low_threshold = low_threshold    # 0.3
        self.allow_low_quality_matches = allow_low_quality_matches

    # 正向传播，match_quality_matrix就是传入的anchor和gt的iou的混淆矩阵
    def __call__(self, match_quality_matrix):
        """
        计算anchors与每个gtboxes匹配的iou最大值，并记录索引，
        iou<low_threshold索引值为-1， low_threshold<=iou<high_threshold索引值为-2
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        # 判断混淆矩阵元素个数是否为零，这里肯定不满足
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        # M x N 的每一列代表一个anchors与所有gt的匹配iou值
        # matched_vals代表每列的最大值，即每个anchors与所有gt匹配的最大iou值
        # matches对应最大值所在的索引
        # 在混淆矩阵的维度0(gt的维度上，即每个anchor与gt的最大iou值)上去求最大值，返回的第一个元素就是他所对应的最大数值，matches为最大值所对应的索引(第几个gt)
        matched_vals, matches = match_quality_matrix.max(dim=0)  # the dimension to reduce.
        # 这里传入的是True
        if self.allow_low_quality_matches:
            # 使用clone的方法clone一份matches
            all_matches = matches.clone()
        else:
            all_matches = None

        # Assign candidate matches with low quality to negative (unassigned) values
        # 计算iou小于low_threshold的索引，得到的结果是一个蒙版，小于这个阈值的设置为true，shape和matched_val一样
        below_low_threshold = matched_vals < self.low_threshold
        # 计算iou在low_threshold与high_threshold之间的索引值，再找到一个0.3-0.7之间的
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold
        )
        # iou小于low_threshold的matches索引置为-1，根据刚才的蒙版赋值，也就是负样本设置为-1
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD  # -1

        # iou在[low_threshold, high_threshold]之间的matches索引置为-2，即丢弃的样本设置为-2
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS    # -2

        # 是否启用匹配正样本的时候，与之匹配iou值最大的anchor也将其设置为正样本，相当于一个样本补充，因为上面的匹配准则可能会导致一个正样本也匹配不到，会导致有的GT被漏掉
        if self.allow_low_quality_matches:
            assert all_matches is not None
            # 传的为原始矩阵，矩阵的索引，过滤后的矩阵
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has highest quality
        # 对于每个gt boxes寻找与其iou最大的anchor，
        # highest_quality_foreach_gt为匹配到的最大iou值
        # 将match_quality_matrix.max在维度１上取最大值，对每个gt与anchor进行最大值iou求取，将求取的最大值赋值给 highest_quality_foreach_gt
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)  # the dimension to reduce.

        # Find highest quality match available, even if it is low, including ties
        # 寻找每个gt boxes与其iou最大的anchor索引，一个gt匹配到的最大iou可能有多个anchor
        # gt_pred_pairs_of_highest_quality = torch.nonzero(
        #     match_quality_matrix == highest_quality_foreach_gt[:, None]
        # )
        # 将match_quality_matrix, highest_quality_foreach_gt可进行对比
        gt_pred_pairs_of_highest_quality = torch.where(
            # 对比两个矩阵，相同的地方为1，不同的地方为0，返回的是相同元素的位置坐标
            torch.eq(match_quality_matrix, highest_quality_foreach_gt[:, None])
        )
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        # gt_pred_pairs_of_highest_quality[:, 0]代表是对应的gt index(不需要)
        # pre_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        # 在维度1上去索引为1的部分
        pre_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        # 保留该anchor匹配gt最大iou的索引，即使iou低于设定的阈值。这里其实是第二条匹配准则，更新matches的索引，把漏掉的gt匹配上去
        matches[pre_inds_to_update] = all_matches[pre_inds_to_update]


def smooth_l1_loss(input, target, beta: float = 1. / 9, size_average: bool = True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    # cond = n < beta
    cond = torch.lt(n, beta)
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()
