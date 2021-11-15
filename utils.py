try:
    import xml.etree.ElementTree as ET
    import os
    import cv2
    import sys
    import shutil
    import imgaug as ia
    import numpy as np
    from imgaug import augmenters as iaa
    from PIL import Image
    import pandas as pd
except:
    print("Please check if all libraries are installed")
    exit()

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
# sys.stderr = Logger('debug.log_file', sys.stderr)
# mkdir
def mkdir(path):
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
def renamesOldName(path, name):
    path = str(path)
    name = str(name)
    fileNameLst = name.split('.')
    filename = fileNameLst[0]
    fileExtension = str("." + fileNameLst[1])
    newname = str("000000" + fileExtension)
    os.renames(path + name, path + newname)
    oldnameLst = [name, newname]
    return oldnameLst
# output file name list from path
def read_fileName_in_path(path):
    files = os.listdir(path)
    files.sort()
    fileNameLst = []
    for file_ in files:
        #    print(path +file_)
        if not os.path.isdir(path + file_):
            f_name = str(file_)
            #        print(f_name)
            fileNameLst.append(f_name)
            # f.write(f_name + '\n')
    return fileNameLst
# bndbox for xml file output
def read_xml_annotation(Path, fileName):
    in_file = open(os.path.join(Path, fileName))
    tree = ET.parse(in_file)
    Path = tree.getroot()
    bndboxlist = []

    for object in Path.findall('object'):
        name = object.find('name').text
        bndbox = object.find('bndbox')
        xmin = bndbox.find('xmin').text
        xmax = bndbox.find('xmax').text
        ymin = bndbox.find('ymin').text
        ymax = bndbox.find('ymax').text
        # print(xmin,ymin,xmax,ymax)
        bndboxlist.append([name, xmin, ymin, xmax, ymax])
        # print(bndboxlist)
    try:
        bndbox = Path.find('object').find('bndbox')
    except:
        print(fileName + "'s bndbox is empty")
        pass
    return bndboxlist
# change the mapping relationship in bndbox
def Category_mapping_with_Hackathon3(bndbox):
    Source_list = [['car', 'bus'], ['bicycle', 'motorbike'], ['person']]
    newbndbox = []
    for lst in bndbox:
        if lst[0] in Source_list[0]:
            lst[0] = 'vechicle'
            newbndbox.append(lst)
        elif lst[0] in Source_list[1]:
            lst[0] = 'bicycle'
            newbndbox.append(lst)
        elif lst[0] in Source_list[2]:
            lst[0] = 'pedestrian'
            newbndbox.append(lst)
    # print(newbndbox)
    return newbndbox
# add '\n' in xml file
def __indent(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            __indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
# crate the xml file (VOC)
def crate_xml_file(write_path,file_name_,new_bndbox,Folder=None,img_path=None):
    mkdir(write_path)
    emptyList = []
    # file = open(write_path + "crate_log.txt",'a')
    if new_bndbox == emptyList:
        # file.write(file_name_ + "'s bndbox is empty")
        # file.write('\n')
        print(file_name_ + "'s bndbox is empty")
        pass
    elif new_bndbox != emptyList:
        # file.write(file_name_ + " : " + str(new_bndbox))
        # file.write('\n')
        print(new_bndbox)
        # print(filename_)
        root = ET.Element('annotation')
        folder = ET.SubElement(root, 'folder')
        if Folder != None:
            folder.text = Folder
        elif Folder == None:
            folder.text = 'default folder'
        filename = ET.SubElement(root, 'filename')
        filename.text = file_name_
        source = ET.SubElement(root, 'source')
        database = ET.SubElement(source, 'database')
        database.text = 'Unknown'
        if img_path != None:
            img = cv2.imread(img_path + file_name_[:-4] + os.path.splitext(os.listdir(img_path)[0])[-1])
            # print(img.shape)
            size = ET.SubElement(root, 'size')
            width = ET.SubElement(size, 'width')
            width.text = str(img.shape[1])
            height = ET.SubElement(size, 'height')
            height.text = str(img.shape[0])
            depth = ET.SubElement(size, 'depth')
            depth.text = str(img.shape[2])
        else:
            pass
        segmented = ET.SubElement(root, 'segmented')
        segmented.text = '0'
        for box in new_bndbox:
            object = ET.SubElement(root, 'object')
            name = ET.SubElement(object, 'name')
            name.text = box[0]
            bndbox = ET.SubElement(object, 'bndbox')
            xmin = ET.SubElement(bndbox,'xmin')
            xmin.text = str(box[1])
            ymin = ET.SubElement(bndbox,'ymin')
            ymin.text = str(box[2])
            xmax = ET.SubElement(bndbox,'xmax')
            xmax.text = str(box[3])
            ymax = ET.SubElement(bndbox,'ymax')
            ymax.text = str(box[4])
        # ET.dump(root)
        tree = ET.ElementTree(root)
        __indent(root)
        tree.write(write_path + file_name_)
# count the number of each label in the xml file
def num_of_xmllabels_read(label_path,filename_):
    name_list = []
    if filename_[-4:] == '.xml':
        in_file = open(os.path.join(label_path, filename_))
        # print(in_file)
        tree = ET.parse(in_file)
        Path = tree.getroot()
        for object in Path.findall('object'):  # 找到Path节点下的所有country节点
            name = object.find('name').text
            name_list.append(name)
    elif filename_[-4:] != '.xml':
        pass
    return name_list
# change the xml file with the new bndbox
def change_bndbox_to_newbndbox(path,write_path,newbndbox,filename,new_filename,file_Prefix=None):
    in_file = open(os.path.join(path, filename))
    tree = ET.parse(in_file)
    elem = tree.find('filename')
    # print(new_filename)
    if file_Prefix == None:
        elem.text = new_filename
    else:
        elem.text = (file_Prefix + new_filename)
    xmlroot = tree.getroot()
    num = 0
    # print(newbndbox)
    for object in xmlroot.findall('object'):
        # print(object)
        bndbox = object.find('bndbox')
        # print(newbndbox)
        new_xmin = newbndbox[num][0]
        new_ymin = newbndbox[num][1]
        new_xmax = newbndbox[num][2]
        new_ymax = newbndbox[num][3]
        print([new_xmin,new_ymin,new_xmax,new_ymax])
        xmin = bndbox.find('xmin')
        xmin.text = str(new_xmin)
        ymin = bndbox.find('ymin')
        ymin.text = str(new_ymin)
        xmax = bndbox.find('xmax')
        xmax.text = str(new_xmax)
        ymax = bndbox.find('ymax')
        ymax.text = str(new_ymax)

        num += 1
    if file_Prefix == None:
        # print(write_path + new_filename)
        tree.write(write_path + new_filename)
        # tree.write()
    else:
        tree.write(write_path + file_Prefix + new_filename)
# data enhancement
def data_enhancement(seq, AUGLOOP, AUGPath, source_img_path, source_xml_path, filename_for_label, file_Prefix, seed=1):
    ia.seed(seed)
    # Specify the path and call the mkdir function to determine and create the path
    AUG_IMG_PATH = AUGPath + file_Prefix + "AUG_IMG\\"
    AUG_XML_PATH = AUGPath + file_Prefix + "AUG_XML\\"
    try:
        shutil.rmtree(AUG_XML_PATH)
        shutil.rmtree(AUG_IMG_PATH)
        print("finish clear file path")
    except:
        pass
    mkdir(AUG_IMG_PATH)
    mkdir(AUG_XML_PATH)
    boxes_img_aug_list = []
    new_bndbox = []
    new_bndbox_list = []
    num = 1
    for filename in filename_for_label:
        new_filename = file_Prefix + "%06d"%num
        bndbox = read_xml_annotation(source_xml_path, filename)
        # copy the xmlfile to aug path
        shutil.copy(os.path.join(source_xml_path, filename), AUG_XML_PATH)
        # copy the picture to aug path
        shutil.copy(os.path.join(source_img_path, filename[:-4] + os.path.splitext(os.listdir(source_img_path)[0])[-1]), AUG_IMG_PATH)
        # rename the copied file and create variables
        Renamesxml = renamesOldName(AUG_XML_PATH, filename)
        Renamesimg = renamesOldName(AUG_IMG_PATH, filename[:-4] + os.path.splitext(os.listdir(source_img_path)[0])[-1])
        xmlNewName,xmlOldname = Renamesxml[1],Renamesxml[0]
        imgNewName,imgOldname = Renamesimg[1],Renamesimg[0]

        for epoch in range(AUGLOOP):
            # Keep the coordinates and the image changing synchronously, rather than randomly changing
            seq_det = seq.to_deterministic()
            # print(imgNewName)
            img = Image.open(os.path.join(AUG_IMG_PATH, imgNewName))
            img = np.asarray(img)
            # bndbox coordinate enhancement
            for i in range(len(bndbox)):
                bbs = ia.BoundingBoxesOnImage([
                    ia.BoundingBox(x1=bndbox[i][1], y1=bndbox[i][2], x2=bndbox[i][3], y2=bndbox[i][4]),
                ], shape=img.shape)

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
            new_image_file = os.path.join(AUG_IMG_PATH, new_filename + os.path.splitext(os.listdir(source_img_path)[0])[-1])
            image_auged = bbs.draw_on_image(image_aug, thickness=0)
            Image.fromarray(image_auged).save(new_image_file)
            print(AUG_IMG_PATH + new_filename + os.path.splitext(os.listdir(source_img_path)[0])[-1])

            # Store the changed XML
            change_bndbox_to_newbndbox(source_xml_path,AUG_XML_PATH,new_bndbox_list,filename,new_filename + ".xml")
            print(AUG_XML_PATH + new_filename + ".xml")
            new_bndbox_list = []
            print("#######################################")

        # remove copied file
        path1 = AUG_XML_PATH + xmlNewName
        path2 = AUG_IMG_PATH + imgNewName
        os.remove(path1)
        os.remove(path2)

        num += 1
# find the label_name in xmlpath
def find_label_in_path(xmlpath,label_name):
    output_label_name_list = []
    label_name = str(label_name)
    file_list = read_fileName_in_path(xmlpath)
    for filename in file_list:
        if filename[-4:] == '.xml':
            # print(filename)
            bndbox = read_xml_annotation(xmlpath,filename)
            for box in bndbox:
                labelname = box[0]
                if label_name == labelname:
                    output_label_name_list.append(filename)
                    # print(filename)
        else:
            pass
    output_label_name_set = set(output_label_name_list)
    final_list = list(output_label_name_set)
    if final_list == []:
        print("error not find this label in path! please chack your input!")
        exit()
    return final_list
# remove from one path to the other one path
def removefile(fromPath,removePath):
    fromLst = os.listdir(fromPath)
    removedLst = os.listdir(removePath)
    num0 = 0
    for fileName in removedLst:
        NewFileName = fileName.split('.')[0] +  "." + fromLst[0].split('.')[1]
        if NewFileName not in fromLst:
            os.remove(removePath + fileName)
            print("already remove " + fileName)
            num0 += 1
    print("already remove " + str(num0) + " files")
    return
# add fileprefix
def add_fileprefox(path,fileprefix):
    filelist = read_fileName_in_path(path)
    for filename in filelist:
        os.renames(path + filename, path + fileprefix + filename)
        print(fileprefix + filename)
    print("Changes are complete! Altogether changed: " + str(len(filelist)))
# Copy path1 file and path2 same name file at the same time
def copy_file_in_two_path(source_path_1,source_path_2,copy_few,write_path):
    source_path_1_list = read_fileName_in_path(source_path_1)
    source_path_2_list = read_fileName_in_path(source_path_2)
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
            cycle_num == len(source_path_1_list) / copy_few
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
            output_path_list.append([A,B])
        except:
            A = output_path_name + source_path_1
            B = output_path_name + source_path_2
            mkdir(A)
            mkdir(B)
            output_path_list.append([A, B])
        index += 1
    # print(output_path_list)
    print("finish crate " + str(len(output_path_list) * 2) + " path")
    # exit()
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
def change_all_label_name(source_path,label_name,write_path=None):
    if write_path == None:
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
            xmlroot = tree.getroot()
            for object in xmlroot.findall('object'):
                name = object.find('name')
                name.text = label_name
            tree.write(output_path + file_name)
            print("already change " + file_name + " in " + output_path)
        elif file_name == 'crate_log.txt':
            print("find the xml crate log")
            pass
# change kitti file to VOC xml file
def kitti_to_VOCxml(kitti_file_path,output_path=None,img_path=None):
    if output_path == None:
        output_path = kitti_file_path + "labels_xml\\"
        mkdir(output_path)
    else:
        mkdir(output_path)
        pass
    for filename in read_fileName_in_path(kitti_file_path):
        if filename.split('.')[1] != 'txt':
            print("Please check the input file type, you need input the .txt file, your input file is " + filename)
            exit()
        else:
            pass
        bndbox = []
        for line in open(kitti_file_path + filename, 'r').readlines():
            line_list = line.split(' ')
            bndbox.append([line_list[0],line_list[4],line_list[5],line_list[6],line_list[7]])
        print(bndbox)
        xml_filename = filename.split('.')[0] + ".xml"
        crate_xml_file(output_path,xml_filename,bndbox,img_path)
        print("already crate " + xml_filename + " in " + output_path)
# Process label into YOLO training
def take_dataset_to_YOLOtrain(Absolute_path,xml_path,img_path,label_list):
    # Absolute_path = os.getcwd()
    # print(Absolute_path)
    YOLO_train_label = open(Absolute_path + "/YOLO_train_label.txt", 'w')
    filename_list = read_fileName_in_path(xml_path)
    # print(filename_list)
    for filename in filename_list:
        bndboxes = read_xml_annotation(xml_path, filename)
        new_bndboxes = []
        YOLO_train_label.write(Absolute_path + "/" + img_path + filename.split('.')[0] + os.path.splitext(os.listdir(img_path)[0])[-1])
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
def filter_the_labels(old_bndboxes,Pixel_values = 40):
    new_bndboxes = []
    index = 0
    for box in old_bndboxes:
        xmin = int(box[1])
        ymin = int(box[2])
        xmax = int(box[3])
        ymax = int(box[4])
        classs_name = str(box[0])
        xmin2xmax = xmax - xmin
        ymin2ymax = ymax - ymin
        if xmin2xmax >= Pixel_values or ymin2ymax >= Pixel_values:
            new_bndboxes.append(box)
        elif xmin2xmax < Pixel_values or ymin2ymax < Pixel_values:
            index += 1
            print("This labels is too small, it will be delete: " + str(box))
    return new_bndboxes,index
# Detect picture brightness
def get_the_image_light(img_path,img_name):
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
    a = 0
    ma = 0
    #np.full 构造一个数组，用指定值填充其元素
    reduce_matrix = np.full((height, width), 128)
    shift_value = gray_img - reduce_matrix
    shift_sum = np.sum(shift_value)
    da = shift_sum / size
    # 计算偏离128的平均偏差
    for i in range(256):
        ma += (abs(i-128-da) * hist[i])
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
    #     pass
    #     print(k)
    #     print(da)
    #     print(img_name + "亮度正常")
    return k,da
# Move too bright or too dark pictures to a new folder
def move_bright_or_dark_img_to_new_folder(img_path,output_path):
    dark_img_list = []
    bright_img_list = []
    for img_name in read_fileName_in_path(img_path):
        brightness,da = get_the_image_light(img_path,img_name)
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
    if dark_img_list  != []:
        dark_path = output_path + "too_dark_img/"
        mkdir(dark_path)
        for filename in dark_img_list:
            shutil.move(img_path + filename,dark_path)
            print("already move the: " + img_path + filename)
        print(len(dark_img_list))
    else:pass
    if bright_img_list != []:
        bright_path = output_path + "too_bright_img/"
        mkdir(bright_path)
        for filename in bright_img_list:
            shutil.move(img_path + filename,bright_path)
            print("already move the: " + img_path + filename)
        print(len(bright_img_list))
    else:pass
    return bright_img_list,dark_img_list
# move file from path to another path
def mvfile_to_path(frompath,topath,file_name_list):
    # mkdir(topath)
    source_file_Suffix = os.path.splitext(os.listdir(frompath)[0])[-1]
    for file_name in file_name_list:
        file_name = file_name[:-4] + source_file_Suffix
        print(file_name)
        shutil.move(frompath + file_name,topath)
def cpfile_to_path(frompath,topath,file_name_list):
    # mkdir(topath)
    source_file_Suffix = os.path.splitext(os.listdir(frompath)[0])[-1]
    for file_name in file_name_list:
        file_name = file_name[:-4] + source_file_Suffix
        print(file_name)
        shutil.copy(frompath + file_name,topath)
# change the specified label name
def change_specified_label_name(source_path,source_label_name,to_label_name,write_path=None):
    if write_path == None:
        output_path = source_path + "changed_label_path/"
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
            xmlroot = tree.getroot()
            for object in xmlroot.findall('object'):
                name = object.find('name')
                if name.text == source_label_name:
                    name.text = to_label_name
                else:pass
            tree.write(output_path + file_name)
            print("already change " + file_name + " in " + output_path)
        elif file_name == 'crate_log.txt':
            print("find the xml crate log")
            pass
# get the bndbox from Hackathon val
def get_box_from_val_path(label_path,file_name):
    file = open(label_path + file_name,'r').readlines()
    bndbox = []
    for line in file:
        line = line.split(' ')
        label_name = line[0]
        xmin,ymin,xmax = line[1],line[2],line[3]
        ymax = line[4][:-1]
        box = [label_name,xmin,ymin,xmax,ymax]
        bndbox.append(box)
    # print(bndbox)
    return bndbox
# VOC to kitti
def VOC_to_kitti(xml_path, write_path):
    mkdir(write_path)
    index = 0
    for filename in read_fileName_in_path(xml_path):
        bndbox = read_xml_annotation(xml_path, filename)
        file = open(write_path + filename[:-4] + ".txt", 'w')
        print(filename + " " + str(bndbox))
        for box in bndbox:
            label_name, xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3], box[4]
            file.write(label_name + " 0.00 0 0.0 " + str(int(xmin)) + ".00 " + str(int(ymin)) + ".00 "
                       + str(int(xmax)) + ".00 " + str(int(ymax)) + ".00 " + " 0.0 0.0 0.0 0.0 0.0 0.0 0.0" + '\n')

            index += 1
    print("already create " + str(index) + " files")

# 按比例分割数据集
