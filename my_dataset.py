# 自定义dataset用于VOC数据集的读取，官方提供的也有，此处是自定义的，怎么自己去写读取数据集的函数。
# 参考文档：https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import numpy as np
from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree

# 创建数据读写函数，必须要有三个默认的函数__init__，__getitem__，__len__
class VOCDataSet(Dataset):
    """读取解析PASCAL VOC2007/2012数据集"""

    # 初始化构造函数，transforms是预处理方法，txt_name表示要读取的文件，训练文件还是测试文件
    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "train.txt"):
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        # 增加容错能力
        # 获取VOC的根目录
        if "VOCdevkit" in voc_root:
            self.root = os.path.join(voc_root, f"VOC{year}")
        else:
            self.root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        # 图片根目录
        self.img_root = os.path.join(self.root, "JPEGImages")
        # 标注信息根目录
        self.annotations_root = os.path.join(self.root, "Annotations")

        # 读取train.txt或者val.txt文件
        txt_path = os.path.join(self.root, "ImageSets", "Main", txt_name)
        assert os.path.exists(txt_path), "not found {} file.".format(txt_name)
        # 通过for循环遍历每一行得到每一张图像的xml文件存到xml_list列表中，里面放的是图像的路径，注意每一个名称后面都有一个换行符，通过line.strip()去掉换行符
        with open(txt_path) as read:
            xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                        for line in read.readlines() if len(line.strip()) > 0]

        self.xml_list = []

        # 检查上面读取的xml文件有没有问题，了解即可，主要是为增加项目的安全性，防止有人更改文件信息
        for xml_path in xml_list:
            if os.path.exists(xml_path) is False:
                print(f"Warning: not found '{xml_path}', skip this annotation file.")
                continue

            # check for targets
            with open(xml_path) as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = self.parse_xml_to_dict(xml)["annotation"]
            if "object" not in data:
                print(f"INFO: no objects in {xml_path}, skip this annotation file.")
                continue

            self.xml_list.append(xml_path)

        assert len(self.xml_list) > 0, "in '{}' file does not find any information.".format(txt_path)

        # 读取类别信息放到self.class_dict中，保存的类别名称和所对应的索引号信息
        json_file = './pascal_voc_classes.json'
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        with open(json_file, 'r') as f:
            self.class_dict = json.load(f)

        # 数据预处理
        self.transforms = transforms

    # 获取图像个数
    def __len__(self):
        return len(self.xml_list)

    # 读取每一张图像，传入的就是一个索引值
    def __getitem__(self, idx):
        # read xml
        # 通过xml列表获取xml文件路径
        xml_path = self.xml_list[idx]
        # 打开xml文件
        with open(xml_path) as fid:
            xml_str = fid.read()
        # 通过etree这个工具来读取xml文件
        xml = etree.fromstring(xml_str)
        # 将读取的xml文件信息传给parse_xml_to_dict，parse_xml_to_dict就是将xml文件解析成字典形式
        data = self.parse_xml_to_dict(xml)["annotation"]
        # 获取文件路径
        img_path = os.path.join(self.img_root, data["filename"])
        # 打开文件
        image = Image.open(img_path)
        # 防止有人篡改文件类型，检查是不是JPEG文件
        if image.format != "JPEG":
            raise ValueError("Image '{}' format not JPEG".format(img_path))
        # 分别定义三个列表用于保存box信息和labels(索引值)以及是否难检测，为零则容易
        boxes = []
        labels = []
        iscrowd = []
        assert "object" in data, "{} lack of object information.".format(xml_path)
        # 遍历每张图片所含的目标信息，获取的是字符型变量，强制转换成浮点型数据
        # data["object"]里面存放的是列表形式的目标信息
        for obj in data["object"]:
            # 获取坐标信息，转为浮点型
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue
            # 将坐标信息放到list里面子再添加到boxes中
            boxes.append([xmin, ymin, xmax, ymax])
            # 获取索引值
            labels.append(self.class_dict[obj["name"]])
            # 获取是否难检测的值
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # 把上面提取的信息转为tensor格式
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        # 当前数据所对应的索引值，就是上面传入的idx
        image_id = torch.tensor([idx])
        # 计算目标框的面积
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # 创建一个目标字典
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        # 是否预处理，注意水平翻转跟分类有所不同，还要翻转bonding box信息
        if self.transforms is not None:
            # 训练的时候传入了这个参数，只是进行了转为张量和水平翻转，可以在transforms.py文件下看到具体实现方式
            image, target = self.transforms(image, target)

        # 返回图像和提取的标签信息
        return image, target

    # 获取高度和宽度，跟上面获取bbox一样，很简单，在使用多GPU训练的时候就需要用到这个方法，如果不提供的话就默认会载入所有的图像来计算高和款，非常的耗时
    def get_height_and_width(self, idx):
        # read xml
        # 获取xml文件
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        # 解析xml文件
        data = self.parse_xml_to_dict(xml)["annotation"]
        # 获取size里面的宽高
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        解析xml文件，将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """
        # 判断是否是最外层，是的话就不等于0
        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        # 套娃，判断子目录是否里面还有子目录
        for child in xml:
            # 递归遍历标签信息
            child_result = self.parse_xml_to_dict(child)
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                # 因为object可能有多个，所以需要放入列表里
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        # 将输入的数据通通过非关键字的形式输入到zip函数进行打包，再将打包的信息转成元组的形式返回。
        return tuple(zip(*batch))

if __name__ == "__main__":
    import transforms
    from draw_box_utils import draw_objs
    from PIL import Image
    import json
    import matplotlib.pyplot as plt
    import torchvision.transforms as ts
    import random

    # read class_indict
    category_index = {}
    try:
        # 读取类别信息
        json_file = open('./pascal_voc_classes.json', 'r')
        class_dict = json.load(json_file)
        # 交换键值对，因为在预测是得到的是索引值而不是类别值，调整json文件的kv值颠倒
        category_index = {str(v): str(k) for k, v in class_dict.items()}
    except Exception as e:
        print(e)
        exit(-1)

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    # 加载数据集
    train_data_set = VOCDataSet("D:\datasets\目标检测", "2012", data_transform["train"], "train.txt")
    # 打印文件个数
    print(len(train_data_set))
    # 随机采用5张图
    for index in random.sample(range(0, len(train_data_set)), k=5):
        # 传入索引
        img, target = train_data_set[index]
        # 转回PIL格式
        img = ts.ToPILImage()(img)
        # 绘制目标框
        plot_img = draw_objs(img,
                             target["boxes"].numpy(),
                             target["labels"].numpy(),
                             np.ones(target["labels"].shape[0]),
                             category_index=category_index,
                             box_thresh=0.5,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)
        plt.imshow(plot_img)
        plt.show()
