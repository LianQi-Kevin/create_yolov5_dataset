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
```
注: windows下运行，如果pycharm报如下错误，则改小batch-size 并调大虚拟内存 参考CSDN
# https://blog.csdn.net/weixin_43959833/article/details/116669523?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link
OSError: [WinError 1455] 页面文件太小，无法完成操作。 Error loading "C:\ProgramData\Anaconda3\envs\yolov5_6\lib\site-packages\torch\lib\caffe2_detectron_ops_gpu.dll" or one of its dependencies.
```



##### 1. 安装环境
```
pip install -r requirements.txt
```
##### 2. 