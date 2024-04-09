import os
import shutil
import sys
import xml.etree.ElementTree as ET
from typing import List

import cv2
import imgaug as ia
import numpy as np
from PIL import Image


# crate debug log file
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger('debug.log', sys.stdout)


# mkdir
def mkdir(path: str):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' Creat successfully')
        return True
    else:
        print(path + ' Dir already exists')
        return False


# rename the old name to 000000
def renamesOldName(path: str, name: str):
    fileExtension = str("." + name.split('.')[1])
    new_name = str("000000" + fileExtension)
    os.renames(path + name, path + new_name)
    return [name, new_name]


# output file name list from path
def read_fileName_in_path(path: str):
    files = os.listdir(path)
    files.sort()
    fileNameLst = []
    for file_ in files:
        #    print(path +file_)
        if not os.path.isdir(path + file_):
            f_name = str(file_)
            #        print(f_name)
            fileNameLst.append(f_name)  # f.write(f_name + '\n')
    return fileNameLst


# bndbox for xml file output
def read_xml_annotation(Path: str, file_name: str):
    in_file = open(os.path.join(Path, file_name))
    tree = ET.parse(in_file)
    Path = tree.getroot()
    bndbox_list = []

    for _object in Path.findall('object'):
        name = _object.find('name').text
        bndbox = _object.find('bndbox')
        xmin = bndbox.find('xmin').text
        xmax = bndbox.find('xmax').text
        ymin = bndbox.find('ymin').text
        ymax = bndbox.find('ymax').text
        # print(xmin,ymin,xmax,ymax)
        bndbox_list.append([name, xmin, ymin, xmax, ymax])
    try:
        Path.find('object').find('bndbox')
    except Exception as _:
        print(file_name + "'s bndbox is empty")
    return bndbox_list


# change the mapping relationship in bndbox
def Category_mapping_with_Hackathon3(bndbox: List[List[str]]):
    Source_list = [['car', 'bus'], ['bicycle', 'motorbike'], ['person']]
    new_bndbox = []
    for lst in bndbox:
        if lst[0] in Source_list[0]:
            lst[0] = 'vechicle'
            new_bndbox.append(lst)
        elif lst[0] in Source_list[1]:
            lst[0] = 'bicycle'
            new_bndbox.append(lst)
        elif lst[0] in Source_list[2]:
            lst[0] = 'pedestrian'
            new_bndbox.append(lst)

    return new_bndbox


# add '\n' in xml file
def __indent(elem: ET.Element, level: int = 0):
    i = "\n" + level * "\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            __indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


# crate the xml file (VOC)
def crate_xml_file(write_path: str, file_name: str, new_bndbox: List[List[str]], Folder: str = None,
                   img_path: str = None):
    mkdir(write_path)
    emptyList = []
    # file = open(write_path + "crate_log.txt",'a')
    if new_bndbox == emptyList:
        # file.write(file_name_ + "'s bndbox is empty")
        # file.write('\n')
        print(file_name + "'s bndbox is empty")
    elif new_bndbox != emptyList:
        # file.write(file_name_ + " : " + str(new_bndbox))
        # file.write('\n')
        print(new_bndbox)
        # print(filename_)
        root = ET.Element('annotation')
        folder = ET.SubElement(root, 'folder')
        folder.text = 'default folder' if Folder is None else Folder
        filename = ET.SubElement(root, 'filename')
        filename.text = file_name
        source = ET.SubElement(root, 'source')
        database = ET.SubElement(source, 'database')
        database.text = 'Unknown'
        if img_path is not None:
            img = cv2.imread(img_path + file_name[:-4] + os.path.splitext(os.listdir(img_path)[0])[-1])
            # print(img.shape)
            size = ET.SubElement(root, 'size')
            width = ET.SubElement(size, 'width')
            width.text = str(img.shape[1])
            height = ET.SubElement(size, 'height')
            height.text = str(img.shape[0])
            depth = ET.SubElement(size, 'depth')
            depth.text = str(img.shape[2])

        segmented = ET.SubElement(root, 'segmented')
        segmented.text = '0'
        for box in new_bndbox:
            _object = ET.SubElement(root, 'object')
            name = ET.SubElement(_object, 'name')
            name.text = box[0]
            bndbox = ET.SubElement(_object, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(box[1])
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(box[2])
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(box[3])
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(box[4])
        # ET.dump(root)
        tree = ET.ElementTree(root)
        __indent(root)
        tree.write(write_path + file_name)


# count the number of each label in the xml file
def num_of_xml_labels_read(label_path: str, file_name: str):
    name_list = []
    if file_name[-4:] == '.xml':
        in_file = open(os.path.join(label_path, file_name))
        # print(in_file)
        tree = ET.parse(in_file)
        Path = tree.getroot()
        for _object in Path.findall('object'):  # 找到Path节点下的所有country节点
            name = _object.find('name').text
            name_list.append(name)
    return name_list


# change the xml file with the new bndbox
def change_bndbox_to_new_bndbox(path: str, write_path: str, new_bndbox: List[List[int]], filename: str,
                                new_filename: str, file_Prefix: str = None):
    in_file = open(os.path.join(path, filename))
    tree = ET.parse(in_file)
    elem = tree.find('filename')
    # print(new_filename)
    if file_Prefix is None:
        elem.text = new_filename
    else:
        elem.text = (file_Prefix + new_filename)
    xml_root = tree.getroot()
    num = 0
    for _object in xml_root.findall('object'):
        bndbox = _object.find('bndbox')
        new_xmin = new_bndbox[num][0]
        new_ymin = new_bndbox[num][1]
        new_xmax = new_bndbox[num][2]
        new_ymax = new_bndbox[num][3]
        print([new_xmin, new_ymin, new_xmax, new_ymax])
        xmin = bndbox.find('xmin')
        xmin.text = str(new_xmin)
        ymin = bndbox.find('ymin')
        ymin.text = str(new_ymin)
        xmax = bndbox.find('xmax')
        xmax.text = str(new_xmax)
        ymax = bndbox.find('ymax')
        ymax.text = str(new_ymax)

        num += 1
    if file_Prefix is None:
        # print(write_path + new_filename)
        tree.write(write_path + new_filename)  # tree.write()
    else:
        tree.write(write_path + file_Prefix + new_filename)


# data enhancement
def data_enhancement(seq, AUG_LOOP, AUGPath: str, source_img_path: str, source_xml_path: str, filename_for_label: str,
                     file_Prefix: str, seed: int = 1):
    ia.seed(seed)
    # Specify the path and call the mkdir function to determine and create the path
    AUG_IMG_PATH = AUGPath + file_Prefix + "AUG_IMG\\"
    AUG_XML_PATH = AUGPath + file_Prefix + "AUG_XML\\"
    try:
        shutil.rmtree(AUG_XML_PATH)
        shutil.rmtree(AUG_IMG_PATH)
        print("finish clear file path")
    except Exception as _:
        pass
    mkdir(AUG_IMG_PATH)
    mkdir(AUG_XML_PATH)
    boxes_img_aug_list = []
    new_bndbox_list = []
    num = 1
    for filename in filename_for_label:
        new_filename = file_Prefix + "%06d" % num
        bndbox = read_xml_annotation(source_xml_path, filename)
        # copy the xml file to aug path
        shutil.copy(os.path.join(source_xml_path, filename), AUG_XML_PATH)
        # copy the picture to aug path
        shutil.copy(os.path.join(source_img_path, filename[:-4] + os.path.splitext(os.listdir(source_img_path)[0])[-1]),
                    AUG_IMG_PATH)
        # rename the copied file and create variables
        renames_xml = renamesOldName(AUG_XML_PATH, filename)
        renames_img = renamesOldName(AUG_IMG_PATH, filename[:-4] + os.path.splitext(os.listdir(source_img_path)[0])[-1])
        xmlNewName, xmlOldName = renames_xml[1], renames_xml[0]
        imgNewName, imgOldName = renames_img[1], renames_img[0]

        for epoch in range(AUG_LOOP):
            # Keep the coordinates and the image changing synchronously, rather than randomly changing
            seq_det = seq.to_deterministic()
            # print(imgNewName)
            img = Image.open(os.path.join(AUG_IMG_PATH, imgNewName))
            img = np.asarray(img)
            # bndbox coordinate enhancement
            bbs = ia.BoundingBoxesOnImage(
                [ia.BoundingBox(x1=bndbox[0][1], y1=bndbox[0][2], x2=bndbox[0][3], y2=bndbox[0][4]), ], shape=img.shape)

            for i in range(len(bndbox)):
                bbs = ia.BoundingBoxesOnImage(
                    [ia.BoundingBox(x1=bndbox[i][1], y1=bndbox[i][2], x2=bndbox[i][3], y2=bndbox[i][4]), ],
                    shape=img.shape)

                bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
                boxes_img_aug_list.append(bbs_aug)

                # new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
                n_x1 = int(max(1, min(img.shape[1], bbs_aug.bounding_boxes[0].x1)))
                n_y1 = int(max(1, min(img.shape[0], bbs_aug.bounding_boxes[0].y1)))
                n_x2 = int(max(1, min(img.shape[1], bbs_aug.bounding_boxes[0].x2)))
                n_y2 = int(max(1, min(img.shape[0], bbs_aug.bounding_boxes[0].y2)))
                if n_x1 == 1 and n_x1 == n_x2:
                    n_x2 += 1
                if n_y1 == 1 and n_y2 == n_y1:
                    n_y2 += 1
                if n_x1 >= n_x2 or n_y1 >= n_y2:
                    print('error', filename)
                new_bndbox_list.append([n_x1, n_y1, n_x2, n_y2])
            print(new_bndbox_list)
            # exit()

            # store the changed picture
            image_aug = seq_det.augment_images([img])[0]
            new_image_file = os.path.join(AUG_IMG_PATH,
                                          new_filename + os.path.splitext(os.listdir(source_img_path)[0])[-1])
            image_augmented = bbs.draw_on_image(image_aug, thickness=0)
            Image.fromarray(image_augmented).save(new_image_file)
            print(AUG_IMG_PATH + new_filename + os.path.splitext(os.listdir(source_img_path)[0])[-1])

            # Store the changed XML
            change_bndbox_to_new_bndbox(source_xml_path, AUG_XML_PATH, new_bndbox_list, filename, new_filename + ".xml")
            print(AUG_XML_PATH + new_filename + ".xml")
            new_bndbox_list = []
            print("#######################################")

        # remove copied file
        path1 = AUG_XML_PATH + xmlNewName
        path2 = AUG_IMG_PATH + imgNewName
        os.remove(path1)
        os.remove(path2)

        num += 1


# find the label_name in xml_path
def find_label_in_path(xml_path: str, label_name: str):
    output_label_name_list = []
    file_list = read_fileName_in_path(xml_path)
    for filename in file_list:
        if filename[-4:] == '.xml':
            # print(filename)
            bndbox = read_xml_annotation(xml_path, filename)
            for box in bndbox:
                if label_name == box[0]:
                    output_label_name_list.append(filename)  # print(filename)

    output_label_name_set = set(output_label_name_list)
    final_list = list(output_label_name_set)
    if not final_list:
        print("error not find this label in path! please check your input!")
        exit()
    return final_list


# remove from one path to the other one path
def remove_file(fromPath: str, removePath: str):
    fromLst = os.listdir(fromPath)
    removedLst = os.listdir(removePath)
    num0 = 0
    for fileName in removedLst:
        NewFileName = fileName.split('.')[0] + "." + fromLst[0].split('.')[1]
        if NewFileName not in fromLst:
            os.remove(removePath + fileName)
            print("already remove " + fileName)
            num0 += 1
    print("already remove " + str(num0) + " files")


# add file-prefix
def add_filePrefix(path: str, file_prefix: str):
    file_list = read_fileName_in_path(path)
    for filename in file_list:
        os.renames(path + filename, path + file_prefix + filename)
        print(file_prefix + filename)
    print("Changes are complete! Altogether changed: " + str(len(file_list)))


# Copy path1 file and path2 same name file at the same time
def copy_file_in_two_path(source_path_1: str, source_path_2: str, copy_few: int, write_path: str):
    source_path_1_list = read_fileName_in_path(source_path_1)
    source_path_2_list = read_fileName_in_path(source_path_2)
    cycle_num = 1
    if len(source_path_1_list) != len(source_path_2_list):
        print("Please check the number of files in the input path! ")
        exit()
    elif len(source_path_1_list) == 0:
        print("Input path is empty, please check your input!")
        exit()
    elif len(source_path_1_list) == len(source_path_2_list):
        # Define the number of files in each group
        if len(source_path_1_list) / copy_few != int(len(source_path_1_list) / copy_few):
            cycle_num = int(len(source_path_1_list) / copy_few) + 1
        elif len(source_path_1_list) / copy_few == int(len(source_path_1_list) / copy_few):
            cycle_num = len(source_path_1_list) / copy_few
    # Traverse to create a new directory
    index = 1
    output_path_list = []
    for a in range(copy_few):
        output_path_name = write_path + write_path.split('\\')[0] + "_" + str(index) + "\\"
        mkdir(output_path_name)
        try:
            A = output_path_name + source_path_1.split('\\')[-2:-1][0] + "\\"
            B = output_path_name + source_path_2.split('\\')[-2:-1][0] + "\\"
            mkdir(A)
            mkdir(B)
            output_path_list.append([A, B])
        except Exception as _:
            A = output_path_name + source_path_1
            B = output_path_name + source_path_2
            mkdir(A)
            mkdir(B)
            output_path_list.append([A, B])
        index += 1
    # print(output_path_list)
    print("finish crate " + str(len(output_path_list) * 2) + " path")

    num0 = 0
    # Create a loop, the number of times is the number of groups
    for d in range(copy_few):
        # Extract the target path in turn
        output_path_1 = output_path_list[d][0]
        output_path_2 = output_path_list[d][1]
        print(output_path_1)
        print(output_path_2)
        # Define file location
        for e in range(cycle_num):
            file_name_1 = source_path_1_list[num0]
            file_name_2 = file_name_1.split('.')[0] + "." + source_path_2_list[0].split('.')[1]
            shutil.copy(os.path.join(source_path_1, file_name_1), output_path_1)
            print("num = " + str(num0 + 1) + " Finish copy " + file_name_1 + " to " + output_path_1)
            shutil.copy(os.path.join(source_path_2, file_name_2), output_path_2)
            print("num = " + str(num0 + 1) + " Finish copy " + file_name_2 + " to " + output_path_2)
            num0 += 1
            if num0 + 1 > len(source_path_1_list):
                break


# Replace all tags to specified tags
def change_all_label_name(source_path: str, label_name: str, write_path: str = None):
    if write_path is None:
        output_path = source_path + "changed_label_path\\"
        mkdir(output_path)
    else:
        output_path = write_path
        mkdir(output_path)
    for file_name in read_fileName_in_path(source_path):
        # print("filename is " + file_name)
        if file_name.split('.')[1] != 'xml' and file_name != 'crate_log.txt':
            print("Please check the input file type, you need input the .xml file, your input file is " + file_name)
            exit()
        elif file_name.split('.')[1] == 'xml':
            in_file = open(os.path.join(source_path, file_name))
            tree = ET.parse(in_file)
            for _object in tree.getroot().findall('object'):
                name = _object.find('name')
                name.text = label_name
            tree.write(output_path + file_name)
            print("already change " + file_name + " in " + output_path)
        elif file_name == 'crate_log.txt':
            print("find the xml crate log")


# change kitti file to VOC xml file
def kitti_to_VOCxml(kitti_file_path: str, output_path: str = None, img_path: str = None):
    if output_path is None:
        output_path = kitti_file_path + "labels_xml\\"
    mkdir(output_path)

    for filename in read_fileName_in_path(kitti_file_path):
        if filename.split('.')[1] != 'txt':
            print("Please check the input file type, you need input the .txt file, your input file is " + filename)
            exit()

        bndbox = []
        for line in open(kitti_file_path + filename, 'r').readlines():
            line_list = line.split(' ')
            bndbox.append([line_list[0], line_list[4], line_list[5], line_list[6], line_list[7]])
        print(bndbox)
        xml_filename = filename.split('.')[0] + ".xml"
        crate_xml_file(output_path, xml_filename, bndbox, img_path)
        print("already crate " + xml_filename + " in " + output_path)


# Process label into YOLO training
def take_dataset_to_YOLO_train(Absolute_path: str, xml_path: str, img_path: str, label_list: List[str]):
    YOLO_train_label = open(Absolute_path + "/YOLO_train_label.txt", 'w')
    filename_list = read_fileName_in_path(xml_path)
    # print(filename_list)
    for filename in filename_list:
        bndboxes = read_xml_annotation(xml_path, filename)
        YOLO_train_label.write(
            Absolute_path + "/" + img_path + filename.split('.')[0] + os.path.splitext(os.listdir(img_path)[0])[-1])
        for bndbox in bndboxes:
            xmin = bndbox[1]
            ymin = bndbox[2]
            xmax = bndbox[3]
            ymax = bndbox[4]
            class_name = bndbox[0]
            class_id = label_list.index(class_name)
            # box = [xmin,ymin,xmax,ymax,class_name]
            input_things = " " + xmin + "," + ymin + "," + xmax + "," + ymax + "," + str(class_id) + " "
            YOLO_train_label.write(input_things)
            print(input_things)
        YOLO_train_label.write('\n')
    print("YOLO train label file create finish!")


# Filter labels smaller than the specified pixel value
def filter_the_labels(old_bndboxes: List[str], Pixel_values: int = 40):
    new_bndboxes = []
    index = 0
    for box in old_bndboxes:
        xmin = int(box[1])
        ymin = int(box[2])
        xmax = int(box[3])
        ymax = int(box[4])
        xmin2xmax = xmax - xmin
        ymin2ymax = ymax - ymin
        if xmin2xmax >= Pixel_values or ymin2ymax >= Pixel_values:
            new_bndboxes.append(box)
        elif xmin2xmax < Pixel_values or ymin2ymax < Pixel_values:
            index += 1
            print("This labels is too small, it will be delete: " + str(box))
    return new_bndboxes, index


# Detect picture brightness
def get_the_image_light(img_path: str, img_name: str):
    img = cv2.imread(img_path + img_name)
    # 把图片转换为单通道的灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取形状以及长宽
    img_shape = gray_img.shape
    height, width = img_shape[0], img_shape[1]
    size = gray_img.size
    # 灰度图的直方图
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    # 计算灰度图像素点偏离均值(128)程序
    ma = 0
    # np.full 构造一个数组，用指定值填充其元素
    reduce_matrix = np.full((height, width), 128)
    shift_value = gray_img - reduce_matrix
    shift_sum = np.sum(shift_value)
    da = shift_sum / size
    # 计算偏离128的平均偏差
    for i in range(256):
        ma += (abs(i - 128 - da) * hist[i])
    m = abs(ma / size)
    # 亮度系数
    k = abs(da) / m
    # print(k)
    # if k[0] > 1:
    #     # 过亮
    #     if da > 0:
    #         print(k)
    #         print(da)
    #         print(img_name + " 过亮")
    #     else:
    #         print(k)
    #         print(da)
    #         print(img_name + " 过暗")
    # else:
    #     print(k)
    #     print(da)
    #     print(img_name + "亮度正常")
    return k, da


# Move too bright or too dark pictures to a new folder
def move_bright_or_dark_img_to_new_folder(img_path: str, output_path: str):
    dark_img_list = []
    bright_img_list = []
    for img_name in read_fileName_in_path(img_path):
        brightness, da = get_the_image_light(img_path, img_name)
        print(brightness)
        # print(da)
        if brightness[0] > 0.96:
            if da > 0:
                bright_img_list.append(img_name)
                print(img_path + img_name + " 过亮")
            else:
                dark_img_list.append(img_name)
                print(img_path + img_name + " 过暗")
        else:
            print(img_path + img_name + " 亮度正常")
    if dark_img_list:
        dark_path = output_path + "too_dark_img/"
        mkdir(dark_path)
        for filename in dark_img_list:
            shutil.move(img_path + filename, dark_path)
            print("already move the: " + img_path + filename)
        print(len(dark_img_list))

    if bright_img_list:
        bright_path = output_path + "too_bright_img/"
        mkdir(bright_path)
        for filename in bright_img_list:
            shutil.move(img_path + filename, bright_path)
            print("already move the: " + img_path + filename)
        print(len(bright_img_list))

    return bright_img_list, dark_img_list


# move file from path to another path
def move_file_to_path(from_path: str, to_path: str, file_name_list: List[str]):
    source_file_Suffix = os.path.splitext(os.listdir(from_path)[0])[-1]
    for file_name in file_name_list:
        file_name = file_name[:-4] + source_file_Suffix
        print(file_name)
        shutil.move(from_path + file_name, to_path)


def copy_file_to_path(from_path: str, to_path: str, file_name_list: List[str]):
    source_file_Suffix = os.path.splitext(os.listdir(from_path)[0])[-1]
    for file_name in file_name_list:
        file_name = file_name[:-4] + source_file_Suffix
        print(file_name)
        shutil.copy(from_path + file_name, to_path)


# change the specified label name
def change_specified_label_name(source_path: str, source_label_name: str, to_label_name: str, write_path: str = None):
    output_path = source_path + "changed_label_path/" if write_path is None else write_path
    mkdir(output_path)
    for file_name in read_fileName_in_path(source_path):
        # print("filename is " + file_name)
        if file_name.split('.')[1] != 'xml' and file_name != 'crate_log.txt':
            print("Please check the input file type, you need input the .xml file, your input file is " + file_name)
            exit()
        elif file_name.split('.')[1] == 'xml':
            in_file = open(os.path.join(source_path, file_name))
            tree = ET.parse(in_file)
            xml_root = tree.getroot()
            for _object in xml_root.findall('object'):
                name = _object.find('name')
                if name.text == source_label_name:
                    name.text = to_label_name

            tree.write(output_path + file_name)
            print("already change " + file_name + " in " + output_path)
        elif file_name == 'crate_log.txt':
            print("find the xml crate log")


# get the bndbox from Hackathon val
def get_box_from_val_path(label_path: str, file_name: str):
    file = open(label_path + file_name, 'r').readlines()
    bndbox = []
    for line in file:
        line = line.split(' ')
        label_name = line[0]
        xmin, ymin, xmax = line[1], line[2], line[3]
        ymax = line[4][:-1]
        box = [label_name, xmin, ymin, xmax, ymax]
        bndbox.append(box)
    # print(bndbox)
    return bndbox


# VOC to kitti
def VOC_to_kitti(xml_path: str, write_path: str):
    mkdir(write_path)
    index = 0
    for filename in read_fileName_in_path(xml_path):
        bndbox = read_xml_annotation(xml_path, filename)
        file = open(write_path + filename[:-4] + ".txt", 'w')
        print(filename + " " + str(bndbox))
        for box in bndbox:
            label_name, xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3], box[4]
            file.write(label_name + " 0.00 0 0.0 " + str(int(xmin)) + ".00 " + str(int(ymin)) + ".00 " + str(
                int(xmax)) + ".00 " + str(int(ymax)) + ".00 " + " 0.0 0.0 0.0 0.0 0.0 0.0 0.0" + '\n')

            index += 1
    print("already create " + str(index) + " files")


# yolo2voc
def xywh2xyxy(xywh: List[float], img_size: List[int]):
    print(xywh, img_size)
    center_X = xywh[0]
    center_y = xywh[1]
    weights = xywh[2]
    height = xywh[3]
    img_weight = img_size[1]
    img_height = img_size[0]
    xmin = str('%.6f' % (max(((center_X * img_weight) - ((weights * img_weight) / 2)), 0)))
    xmax = str('%.6f' % (min(((center_X * img_weight) + ((weights * img_weight) / 2)), img_weight)))
    ymin = str('%.6f' % (max(((center_y * img_height) - ((height * img_height) / 2)), 0)))
    ymax = str('%.6f' % (min(((center_y * img_height) + ((height * img_height) / 2)), img_height)))
    print(xmin, ymin, xmax, ymax)
    return xmin, ymin, xmax, ymax


def yolo2voc(yolo_label_dir: str, output_dir: str, file_name: str, img_dir: str, class_list: List[str]):
    with open(yolo_label_dir + file_name, "r") as txt_f:
        bndboxes = []
        for line in txt_f.readlines():
            if line != "":
                line_lst = line.split(" ")
                class_name = class_list[int(line_lst[0])]
                center_X = float(line_lst[1])
                center_y = float(line_lst[2])
                weights = float(line_lst[3])
                height = float(line_lst[4])
                xywh = [center_X, center_y, weights, height]
                img_name = file_name.split(".")[0] + "." + read_fileName_in_path(img_dir)[0].split(".")[1]
                print(img_name)
                img = cv2.imread(img_dir + img_name)
                img_height, img_weight = img.shape[0], img.shape[1]
                img_size = [img_height, img_weight]
                xmin, ymin, xmax, ymax = xywh2xyxy(xywh, img_size)
                bndbox = [class_name, xmin, ymin, xmax, ymax]
                bndboxes.append(bndbox)
        print(bndboxes)
        crate_xml_file(output_dir, file_name.split(".")[0] + ".xml", bndboxes)
