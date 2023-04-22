# 以mobile_v2为骨干网络进行训练，不推荐
import os
import datetime

import torch
import torchvision

import transforms
from network_files import FasterRCNN, AnchorsGenerator
from backbone import MobileNetV2, vgg
from my_dataset import VOCDataSet
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import train_eval_utils as utils


# 创建网络
def create_model(num_classes):
    # https://download.pytorch.org/models/vgg16-397923af.pth
    # 如果使用vgg16的话就下载对应预训练权重并取消下面注释，接着把mobilenetv2模型对应的两行代码注释掉
    # vgg_feature = vgg(model_name="vgg16", weights_path="./backbone/vgg16.pth").features
    # backbone = torch.nn.Sequential(*list(vgg_feature._modules.values())[:-1])  # 删除features中最后一个Maxpool层
    # backbone.out_channels = 512

    # https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    backbone = MobileNetV2(weights_path="D:\program\Object_detection\\faster_rcnn\\faster_rcnn\\backbone\mobilenet_v2.pth").features
    backbone.out_channels = 1280  # 设置对应backbone输出特征矩阵的channels
    # 对于mobile而言，因为他只有一个预测层，我们在他的预测层上负责预测32, 64, 128, 256, 512一系列尺寸，比例是0.5, 1.0, 2.0,即每个滑动窗口都会预测15个anchor一系列对应参数,预测大小不同的目标
    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),  # 比原文里面多了32和64小尺寸的预测，注意这是个元组类型，后面逗号不能少
                                        aspect_ratios=((0.5, 1.0, 2.0),))
    # featmap_names就是faster_rcnn里面提到的有序字典，mobile_v2只有一个，resnet50有5个，用于输出不同特征的特征矩阵
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行roi pooling
                                                    output_size=[7, 7],  # roi_pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率，因为需要将每个建议狂缩放到7*7大小，所以涉及到采样率

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model


def main():
    # 获取设备信息
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # 用来保存coco_info的文件
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")
    # 图像预处理函数，需要注意标签的转换，对原图进行任何尺寸和方向上的改变都要对标签进行转换。
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }
    # VOC数据集目录所在位置，根据自己的VOC数据集所在位置进行设置
    VOC_root = "D:\datasets\目标检测"

    # 训练的小技巧，按照图片的相似高宽比组成batch
    aspect_ratio_group_factor = 3
    batch_size = 6
    amp = False  # 是否使用混合精度训练，需要GPU支持

    # 检查数据集路径
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    # 通过自己创建的 VOCDataSet读取数据
    train_dataset = VOCDataSet(VOC_root, "2012", data_transform["train"], "train.txt")
    train_sampler = None

    # 是否按图片相似高宽比采样图片组成batch
    # 使用的话能够减小训练时所需GPU显存，默认使用
    if aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # 统计所有图像高宽比例在bins区间中的位置索引
        group_ids = create_aspect_ratio_groups(train_dataset, k=aspect_ratio_group_factor)
        # 每个batch图片从同一高宽比例区间中取
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size)

    # 多线程，windows不支持
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch，自定义数据集的函数返回的是元组
    if train_sampler:
        # 如果按照图片高宽比采样图片，dataloader中需要使用batch_sampler。在这里自己定义了一个collate_fn方法，如过不定义这个方法则默认将数据通过torch.stack()方法对数据进行简单的拼接得到一个个tensor,分类里面可以这样做，因为只有tensor
        # 但是这里并不是一个个的tensor，返回的是tuple类型，如果采用默认的方法肯定会出错，所以要自定义collate_fn函数，将图像信息和标注信息分别放到两个不同的元组中
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)

    # 加载测试数据集
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    val_dataset = VOCDataSet(VOC_root, "2012", data_transform["val"], "val.txt")
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=val_dataset.collate_fn)

    # 创建模型，类别为21是因为背景也算一个
    model = create_model(num_classes=21)
    # print(model)

    # 模型放到指定设备上
    model.to(device)

    # 混合精度amp
    scaler = torch.cuda.amp.GradScaler() if amp else None

    train_loss = []
    learning_rate = []
    val_map = []

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  first frozen backbone and train 5 epochs                   #
    #  首先冻结前置特征提取网络权重（backbone），训练rpn以及最终预测网络部分 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    for param in model.backbone.parameters():
        param.requires_grad = False

    # 定义优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # 先通过5个人epoch微调
    init_epochs = 5
    for epoch in range(init_epochs):
        # train for one epoch, printing every 10 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=50,
                                              warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # evaluate on the test dataset
        coco_info = utils.evaluate(model, val_data_loader, device=device)

        # write into txt
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # pascal mAP

    # 保存模型
    torch.save(model.state_dict(), "./save_weights/pretrain.pth")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  second unfrozen backbone and train all network     #
    #  解冻前置特征提取网络权重（backbone），接着训练整个网络权重  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # 训练第二阶段，训练整个网络
    # 冻结backbone部分底层权重
    for name, parameter in model.backbone.named_parameters():
        split_name = name.split(".")[0]
        if split_name in ["0", "1", "2", "3"]:
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True

    # 定义优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # 设置学习率路线
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.33)
    # 训练次数
    num_epochs = 20
    for epoch in range(init_epochs, num_epochs + init_epochs, 1):
        # train for one epoch, printing every 50 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=50,
                                              warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # 更新学习率
        lr_scheduler.step()

        # 测试模型
        coco_info = utils.evaluate(model, val_data_loader, device=device)

        # write into txt
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        # mAP
        val_map.append(coco_info[1])

        # save weights
        # 仅保存最后5个epoch的权重
        if epoch in range(num_epochs + init_epochs)[-5:]:
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            torch.save(save_files, "./save_weights/mobile-model-{}.pth".format(epoch))

    # 绘制损失和学习率曲线
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # 绘制mAP曲线
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


if __name__ == "__main__":
    main()
