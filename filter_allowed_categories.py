import datetime

from ruamel import yaml
import pandas as pd

from utils import *


def VOC_to_yolo(xml_path: str, img_Path: str, write_path: str, _allow_sort: List[str]):
    # mkdir(write_path)
    index = 0
    for file_name in read_fileName_in_path(xml_path):
        _bndbox = read_xml_annotation(xml_path, file_name)
        with open(write_path + file_name[:-4] + ".txt", 'w') as f:
            print(file_name + " " + str(_bndbox))
            img_name = file_name[:-4] + os.path.splitext(os.listdir(img_Path)[0])[-1]
            img = Image.open(img_Path + img_name)
            img_width, img_height = img.size
            # print(img_width,img_height)
            for box in _bndbox:
                label_num = str(_allow_sort.index(box[0]))
                xmin, ymin, xmax, ymax = float(box[1]), float(box[2]), float(box[3]), float(box[4])
                # 筛选过小的标签
                if xmax - xmin <= 20 or ymax - ymin <= 20:
                    break
                else:
                    # 计算bndbox中心点和大小然后保留6位小数并归一化处理
                    box_width = str('%.6f' % ((xmax - xmin) / img_width))
                    box_height = str('%.6f' % ((ymax - ymin) / img_height))
                    center_x = str('%.6f' % ((xmin + ((xmax - xmin) / 2)) / img_width))
                    center_y = str('%.6f' % ((ymin + ((ymax - ymin) / 2)) / img_height))
                    write_thi = label_num + " " + center_x + " " + center_y + " " + box_width + " " + box_height + "\r"
                    print(write_thi)
                    f.write(write_thi)
            index += 1
    print("already create " + str(index) + " files")


def filter_allowed_categories(XmlFileDir, s_xml_dir, s_img_dir, allow_list, ImgFileDir, _folder: str = None):
    for source_xml_file_name in read_fileName_in_path(XmlFileDir):
        new_bndbox = []
        _bndboxes = read_xml_annotation(XmlFileDir, source_xml_file_name)
        for _bndbox in _bndboxes:
            if _bndbox[0] in allow_list:
                new_bndbox.append(_bndbox)
        print(new_bndbox)
        crate_xml_file(s_xml_dir, source_xml_file_name, new_bndbox, _folder, ImgFileDir)
    copy_file_to_path(ImgFileDir, s_img_dir, read_fileName_in_path(s_xml_dir))


def get_sort_num(xml_file_dir: str):
    label_list = []
    for file_name in read_fileName_in_path(xml_file_dir):
        for name in num_of_xml_labels_read(xml_file_dir, file_name):
            label_list.append(name)
    final_list = pd.value_counts(label_list)
    label_list = list(set(label_list))
    _sort_dict = {}
    for _sort in label_list:
        _sort_dict[_sort] = final_list[_sort]
    return _sort_dict


# VOC标签位置
VOCxml_file_dir = "7th_official_data/labels/"
# 图片位置
img_file_dir = "7th_official_data/images/"
# 输出文件夹位置
output_dir = "./dataset/"
# 比例系数
scale = 0.9
# 允许的标签类别
allow_sort = ["bottle", "banana", "CARDBOARD"]
# 备注标签
folder = "7th_official"
# 训练时使用的预训练模型 yolov5m/s/x/l/n/m6/s6/x6/l6/n6 生成训练命令时使用，cfg文件会生成全部版本的
model_name = "yolov5s"

# 支持的模型类别
cfg = ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x", "yolov5n6", "yolov5s6", "yolov5m6", "yolov5l6",
       "yolov5x6"]
# 今天的日期
today = datetime.date.today()
formatted_today = today.strftime('%y%m%d')

if model_name not in cfg:
    print("Please check model name")
    exit()

# 定义输出文件夹
if str(output_dir)[-1] != "/":
    output_dir = output_dir + "/"
output_dir = output_dir + "dataset_" + formatted_today + "_" + folder + "/"
# source
source_img = output_dir + "source/img/"
source_xml = output_dir + "source/xml/"
# images
images_train = output_dir + "images/train/"
images_val = output_dir + "images/val/"
# labels
labels_train = output_dir + "labels/train/"
labels_val = output_dir + "labels/val/"
# labels_xml
labels_xml_train = output_dir + "labels_xml/train/"
labels_xml_val = output_dir + "labels_xml/val/"
# cfg&data
cfg_dir = output_dir + "cfg/"
data_dir = output_dir + "data/"

# 创建输出文件夹结构
mkdir(source_img)
mkdir(source_xml)
mkdir(images_train)
mkdir(images_val)
mkdir(labels_train)
mkdir(labels_val)
mkdir(labels_xml_train)
mkdir(labels_xml_val)
mkdir(cfg_dir)
mkdir(data_dir)

# 移动允许类别的图片到source文件夹下
filter_allowed_categories(VOCxml_file_dir, source_xml, source_img, allow_sort, img_file_dir, folder)

# 获取标签数量字典
for i in range(len(allow_sort)):
    sort_dict = get_sort_num(source_xml)
    # print(sort_dict)

    # 取出字典中值最小的标签
    min_sort = min(sort_dict, key=sort_dict.get)
    min_value = sort_dict[min_sort]
    # print(min_sort)
    # print(min_value)

    mv_xml_num = int(min_value * (1 - scale))
    print("will mv " + str(mv_xml_num) + " files to " + str(labels_xml_val))
    cp_file_list_train = []
    cp_file_list_val = []
    for filename in read_fileName_in_path(source_xml):
        bndboxes = read_xml_annotation(source_xml, filename)
        for bndbox in bndboxes:
            if bndbox[0] == min_sort:
                if len(cp_file_list_val) >= mv_xml_num:
                    cp_file_list_train.append(filename)
                else:
                    cp_file_list_val.append(filename)
            cp_file_list_train = list(set(cp_file_list_train))
            cp_file_list_val = list(set(cp_file_list_val))
    print(cp_file_list_val)
    print(cp_file_list_train)

    for a in cp_file_list_val:
        if a in cp_file_list_train:
            num = cp_file_list_train.index(a)
            cp_file_list_train.pop(num)

    move_file_to_path(source_xml, labels_xml_val, cp_file_list_val)
    move_file_to_path(source_xml, labels_xml_train, cp_file_list_train)
    # break

move_file_to_path(source_img, images_train, read_fileName_in_path(labels_xml_train))
move_file_to_path(source_img, images_val, read_fileName_in_path(labels_xml_val))
VOC_to_yolo(labels_xml_train, images_train, labels_train, allow_sort)
VOC_to_yolo(labels_xml_val, images_val, labels_val, allow_sort)

# 创建训练配置文件
# data文件
basic_data = open("resource/data/basic_train_data.yaml", "r", encoding='utf-8')
basic_yaml = yaml.load(basic_data.read(), Loader=yaml.Loader)
# print(basic_yaml)
basic_yaml["train"] = images_train
basic_yaml["val"] = images_val
basic_yaml["nc"] = len(allow_sort)
basic_yaml["names"] = allow_sort
data_w = open(data_dir + folder + ".yaml", 'w')
yaml.dump(basic_yaml, data_w)
data_w.close()
# cfg文件
for sort in cfg:
    basic_cfg = open("resource/cfg/" + sort + ".yaml", "r", encoding='utf-8')
    basic_yaml = yaml.load(basic_cfg.read(), Loader=yaml.Loader)
    # print(basic_yaml)
    basic_yaml["nc"] = len(allow_sort)
    data_w = open(cfg_dir + sort + ".yaml", 'w')
    yaml.dump(basic_yaml, data_w)
    data_w.close()

# 输出训练命令
# $ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
#                                          yolov5m                                40
#                                          yolov5l                                24
#                                          yolov5x                                16
train_command = "python train.py --data " + data_dir + folder + ".yaml --cfg " + \
                cfg_dir + model_name + ".yaml --weights " + model_name + ".pt " \
                                                                         "--batch-size 48 --epochs 500"
file = open(output_dir + "train_command.txt", 'w')
file.write(train_command)
file.close()
print(train_command)
