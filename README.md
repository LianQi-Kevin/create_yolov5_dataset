### 从原始数据集筛选标签类型并创建yolov5训练所需格式
标签采用VOC的xml格式\
图片为jpg格式\
按标签类型比例分割数据集\
按照yolov5需求输出\
按需求创建训练需要的yaml文件  并输出训练命令
```
输出的数据集结构: 
dataset
    ├─cfg
    ├─data
    ├─images
    │  ├─train
    │  └─val
    ├─labels
    │  ├─train
    │  └─val
    ├─labels_xml
    │  ├─train
    │  └─val
    └─source
        ├─img
        └─xml
```  

##### 1. 安装环境
```
pip install -r requirements.txt
```
##### 2. 修改参数
```
# VOC标签位置
VOCxml_file_dir = "basic_dataset/labels_xml/"
# 图片位置 图片格式推荐为jpg，png亦可
img_file_dir = "basic_dataset/images/"
# 输出文件夹位置
output_dir = "./"
# 比例系数
scale = 0.9
# 允许的标签类别 用来筛选标签类型 yolo的类别编号也使用此列表
allow_sort = ["cat", "dog", "horse","person"]
# 备注标签
Floder = "example"
# 训练时使用的预训练模型 yolov5m/s/x/l/n 生成训练命令时使用 cfg文件会生成全部版本的
model_name = "yolov5s"
```

##### 3. 备注
###### 1. windows下运行，如果pycharm报如下错误，则改小batch-size 并调大虚拟内存 参考CSDN
``` 
# https://blog.csdn.net/weixin_43959833/article/details/116669523?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link
OSError: [WinError 1455] 页面文件太小，无法完成操作。 Error loading "C:\ProgramData\Anaconda3\envs\yolov5_6\lib\site-packages\torch\lib\caffe2_detectron_ops_gpu.dll" or one of its dependencies.
```
###### 2. 数据集种允许类别的数量请不要过小，否则可能会报错
###### 3. VOCxml的格式如下
```
<annotation>
	<folder>VOC2011</folder>
	<filename>2007_000549.jpg</filename>
	<source>
		<database>The VOC2007 Database</database>
		<annotation>PASCAL VOC2007</annotation>
		<image>flickr</image>
	</source>
	<size>
		<width>375</width>
		<height>500</height>
		<depth>3</depth>
	</size>
	<segmented>1</segmented>
	<object>
		<name>cat</name>
		<pose>Unspecified</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>1</xmin>
			<ymin>49</ymin>
			<xmax>341</xmax>
			<ymax>499</ymax>
		</bndbox>
	</object>
</annotation>
```
###### 4. utils.py文件还中有一些并未被使用的函数，具体用法请自行查看代码