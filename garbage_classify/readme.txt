##########################################
目录结构：
│  readme.txt
│
├─src  				(保存py文件与训练好的模型)
│  │  image_process.py	              （图像工程，将训练集随机分类制作验证集）
│  │  train.py		              （训练文件，网络结构为resnet50V2和自加结构进行迁移学习，准确率95%）
│  │  test.py		                (测试文件，使用test_picture文件夹下十六张图片进行测试)
│  │  inception_resnet_v2.py	              （训练文件，网络结构为inception_resnet_v2和自加结构进行迁移学习，准确率93%）
│  │  inceptionV3.py	              （训练文件，网络结构为inceptionV3和自加结构进行迁移学习，准确率95%）
│  │  resnet50.py		              （训练文件，网络结构为RESTNET50和自加结构进行迁移学习，准确率89%）
│  │  cnn_1.py		              （训练文件，网络结构自加结构，准确率75%）
│  │  cnn_2.py.py		              （训练文件，网络结构自加结构，准确率72%）
 |   |
│  └─模型			              （保存生成的.hdf5模型文件数字为epoch+val_accuracy）
│          cnn2_weights-improvement-20-0.72.hdf5
│          InceptionResNetV2_weights-improvement-26-0.93.hdf5
│          InceptionV3_weights-improvement-99-0.95.hdf5
│          ResNet50V2_weights-improvement-25-0.95.hdf5
│          resnet_50_weights-improvement-190-0.89.hdf5
├─test_picture		              （测试集）
└─垃圾分类		              （数据集）
    ├─train			              （训练集）
    │  ├─glass
    │  ├─metal
    │  ├─paper
    │  └─plastic
    └─validation		             （验证集）
        ├─glass
        ├─metal
        ├─paper
        └─plastic
##########################################
运行方式：
（第一步可省略，以分好数据集）
1.按照文件目录进行创建相应文件夹（validation文件夹不需要创建），数据集修改为中文名称分四类全部放在train中，运行image_process.py产生验证集和测试集。
2.运行train.py进行训练。
3.在test.py中首先指定模型路径再运行test.py进行验证。


训练--验证--测试




