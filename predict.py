import os
import time
import json

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2
from draw_box_utils import draw_objs


# 创建模型,跟训练的时候创建模型是一样的，具体后面再介绍网络的时候会详细说明每句话的意思，这里仅仅看下怎么测试
def create_model(num_classes):
    # 训练的时候没如果有使用冻结，这里测试的时候也不要使用冻结
    # mobileNetv2+faster_RCNN
    backbone = MobileNetV2().features
    backbone.out_channels = 1280

    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=[7, 7],
                                                    sampling_ratio=2)

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    # 注意，这里的norm_layer要和训练脚本中保持一致
    # backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    # model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model

# 测试网络的预测时间用的，不用管
def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # 获取设备，cuda还是cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 创建模型
    model = create_model(num_classes=21)

    # 加载权重文件，注意这里加载权重文件的方式，之前是按照字典的方式保存的，这里也要按照之前保存的方式加载出来
    weights_path = "./save_weights/resNetFpn-model-24.pth"
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    checkpoints = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoints["model"])
    # model.load_state_dict(torch.load(weights_path, map_location='cpu')["model"])

    model.to(device)

    # 类别信息
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    # 键值对对调一下
    category_index = {str(v): str(k) for k, v in class_dict.items()}

    # 加载测试图片
    original_img = Image.open("./img_1.png")

    # Image打开的是PIL文件，转为tensor
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # 加一个batch维度
    img = torch.unsqueeze(img, dim=0)

    # 进入验证模式
    model.eval()
    with torch.no_grad():
        # 先对网络进行初始化，不用管，随机创建一张图像
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        # 测测试预测时间
        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        # 预测结果转为numpy
        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        # 绘制结果
        plot_img = draw_objs(original_img,
                             predict_boxes,
                             predict_classes,
                             predict_scores,
                             category_index=category_index,
                             box_thresh=0.5,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)
        # 显示测试结果
        plt.imshow(plot_img)
        plt.show()
        # 保存预测的图片结果
        plot_img.save("test_result_mobile.jpg")


if __name__ == '__main__':
    main()
