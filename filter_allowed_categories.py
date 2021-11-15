from ruamel import yaml
import datetime
from utils import *

'''
标签采用VOC的xml格式
图片为jpg格式
按标签类型比例分割数据集

按照yolov5需求输出，标签自动转换为kitti格式
按需求创建训练需要的yaml文件  并输出训练命令

─── dataset
    ├──source
        ├── img
        └── xml
    ├── images
        ├── train
        └── val
    ├── labels
        ├── train
        └── val
    └── labels_xml
        ├── train
        └── val
        
注: 如果pycharm报错
OSError: [WinError 1455] 页面文件太小，无法完成操作。 Error loading "C:\ProgramData\Anaconda3\envs\yolov5_6\lib\site-packages\torch\lib\caffe2_detectron_ops_gpu.dll" or one of its depe
ndencies.
则改小batch-size 并调大虚拟内存
https://blog.csdn.net/weixin_43959833/article/details/116669523?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link
'''


def VOC_to_yolo(xml_path, img_Path, write_path, allow_sort):
    # mkdir(write_path)
    index = 0
    for filename in read_fileName_in_path(xml_path):
        bndbox = read_xml_annotation(xml_path, filename)
        file = open(write_path + filename[:-4] + ".txt", 'w')
        print(filename + " " + str(bndbox))
        img_name = filename[:-4] + os.path.splitext(os.listdir(img_Path)[0])[-1]
        img = Image.open(img_Path + img_name)
        img_width, img_height = img.size
        # print(img_width,img_height)
        for box in bndbox:
            label_num = str(allow_sort.index(box[0]))
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
                file.write(write_thi)
        file.close()
        index += 1
    print("already create " + str(index) + " files")


def filter_allowed_categories(XmlFileDir, sxml_dir, simg_dir, allow_list, ImgFileDir, Floder=None):
    for source_xmlfilename in read_fileName_in_path(XmlFileDir):
        new_bndbox = []
        bndboxs = read_xml_annotation(XmlFileDir, source_xmlfilename)
        for bndbox in bndboxs:
            if bndbox[0] in allow_list:
                new_bndbox.append(bndbox)
        print(new_bndbox)
        crate_xml_file(sxml_dir, source_xmlfilename, new_bndbox, Floder, ImgFileDir)
    cpfile_to_path(ImgFileDir, simg_dir, read_fileName_in_path(sxml_dir))


def get_sort_num(xml_file_dir, allow_sort):
    label_list = []
    for filename in read_fileName_in_path(xml_file_dir):
        for name in num_of_xmllabels_read(xml_file_dir, filename):
            label_list.append(name)
    final_list = pd.value_counts(label_list)
    label_list = list(set(label_list))
    sort_dict = {}
    for sort in label_list:
        sort_dict[sort] = final_list[sort]
    return sort_dict


# VOC标签位置
VOCxml_file_dir = "../Dataset/final_dataset/xml/"
# 图片位置
img_file_dir = "../Dataset/final_dataset/img/"
# 输出文件夹位置
output_dir = "./"
# 比例系数
scale = 0.9
# 允许的标签类别
allow_sort = ["cat", "dog", "horse","person"]
# 备注标签
Floder = "H4animal"
# 训练时使用的预训练模型 yolov5m/s/x/l/n
model_name = "yolov5s"
# 支持的模型类别
cfg = ["yolov5n","yolov5s","yolov5m","yolov5l","yolov5x","yolov5n6","yolov5s6","yolov5m6","yolov5l6","yolov5x6"]
# 今天的日期
today=datetime.date.today()
formatted_today=today.strftime('%y%m%d')

if model_name not in cfg:
    print("Please check model name")
    exit()

# 定义输出文件夹
if str(output_dir)[-1] != "/":
    output_dir = output_dir + "/"
output_dir = output_dir + "dataset_" + formatted_today + "_" + Floder + "/"
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
filter_allowed_categories(VOCxml_file_dir, source_xml, source_img, allow_sort, img_file_dir, Floder)

# 获取标签数量字典
for i in range(len(allow_sort)):
    sort_dict = get_sort_num(source_xml, allow_sort)
    print(sort_dict)

    # 取出字典中值最小的标签
    min_sort = min(sort_dict, key=sort_dict.get)
    min_value = sort_dict[min_sort]
    print(min_sort)
    print(min_value)

    mv_xml_num = int(min_value * (1 - scale))
    print("will mv " + str(mv_xml_num) + " files to " + str(labels_xml_val))
    cp_file_list_train = []
    cp_file_list_val = []
    for filename in read_fileName_in_path(source_xml):
        bndboxs = read_xml_annotation(source_xml, filename)
        # rand_num = float(str(np.random.uniform(0, 1))[0:3])
        for bndbox in bndboxs:
            if bndbox[0] == min_sort:
                if len(cp_file_list_val) >= mv_xml_num:
                    cp_file_list_train.append(filename)
                else:
                    cp_file_list_val.append(filename)
            cp_file_list_train = list(set(cp_file_list_train))
            cp_file_list_val = list(set(cp_file_list_val))
    print(cp_file_list_val)
    print(len(cp_file_list_val))
    print(cp_file_list_train)
    print(len(cp_file_list_train))

    for a in cp_file_list_val:
        if a in cp_file_list_train:
            num = cp_file_list_train.index(a)
            cp_file_list_train.pop(num)

    mvfile_to_path(source_xml, labels_xml_val, cp_file_list_val)
    mvfile_to_path(source_xml, labels_xml_train, cp_file_list_train)
    # break

mvfile_to_path(source_img, images_train, read_fileName_in_path(labels_xml_train))
mvfile_to_path(source_img, images_val, read_fileName_in_path(labels_xml_val))
VOC_to_yolo(labels_xml_train, images_train, labels_train, allow_sort)
VOC_to_yolo(labels_xml_val, images_val, labels_val, allow_sort)

# 创建训练配置文件
# data文件
basic_data = open("resource/data/basic_train_data.yaml", "r", encoding='utf-8')
basic_yaml = yaml.load(basic_data.read(), Loader=yaml.Loader)
print(basic_yaml)
basic_yaml["train"] = images_train
basic_yaml["val"] = images_val
basic_yaml["nc"] = len(allow_sort)
basic_yaml["names"] = allow_sort
data_w = open(data_dir + Floder + ".yaml", 'w')
yaml.dump(basic_yaml, data_w)
data_w.close()
# cfg文件
for sort in cfg:
    basic_cfg = open("resource/cfg/" + sort + ".yaml", "r", encoding='utf-8')
    basic_yaml = yaml.load(basic_cfg.read(), Loader=yaml.Loader)
    print(basic_yaml)
    basic_yaml["nc"] = len(allow_sort)
    data_w = open(cfg_dir +  sort + ".yaml", 'w')
    yaml.dump(basic_yaml, data_w)
    data_w.close()

# 输出训练命令
# $ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
#                                          yolov5m                                40
#                                          yolov5l                                24
#                                          yolov5x                                16
train_command = "python train.py --data " + data_dir + Floder + ".yaml --cfg " + \
                cfg_dir + model_name + ".yaml --weights "  + model_name + ".pt " \
                "--batch-size 48 --epochs 500"
file = open(output_dir + "train_command.txt", 'w')
file.write(train_command)
file.close()
print(train_command)
