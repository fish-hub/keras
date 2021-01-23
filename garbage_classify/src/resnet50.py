import yaml
import sys,time
import string
import json
import cv2
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.models import model_from_yaml
import pylab
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np
import os
import scipy.io as io
from multiprocessing import Pool
from tensorflow.python.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from tensorflow.python.keras.preprocessing import image
from PIL import Image
import matplotlib.image as mpimg
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras import layers
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.applications.xception  import Xception
from tensorflow.python.keras.models import Model, Sequential,load_model
from tensorflow.python.keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D,Flatten,AveragePooling2D,add,BatchNormalization,Convolution2D,ZeroPadding2D,Reshape,Activation, Dense,Lambda,Conv2D,MaxPool2D, Flatten, Dropout,MaxPooling2D,Dense,concatenate,GlobalAveragePooling2D
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import backend, initializers,regularizers,callbacks,Input,optimizers
image_lengh,image_width = 0,0
from keras import layers,models,applications,optimizers
def get_image_size(filename):
    #查看图像尺寸：
    from PIL import Image
    img = Image.open(filename)
    imgSize = img.size
    image_lengh = imgSize[0]
    image_width = imgSize[1]
    print("图片尺寸为：",imgSize)
    return image_lengh,image_width

#图像处理
def image_process(batch_size,train_dir,test_dir,image_lengh,image_width):
    datagen_train = ImageDataGenerator(
                          rescale=1./255  ,
                          rotation_range=180,
                          width_shift_range=0.1,
                          height_shift_range=0.1,
                          shear_range=0.1,
                          zoom_range=[0.9, 1.5],
                          horizontal_flip=True,
                          vertical_flip=True,
                          fill_mode='nearest'
                         )
    datagen_test = ImageDataGenerator(rescale=1./255  ,
                          rotation_range=180,
                          width_shift_range=0.1,
                          height_shift_range=0.1,
                          shear_range=0.1,
                          zoom_range=[0.9, 1.5],
                          horizontal_flip=True,
                          vertical_flip=True,
                          fill_mode='nearest')
    generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                    target_size=(image_lengh,image_width),
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    class_mode='categorical')

    generator_test = datagen_test.flow_from_directory(directory=test_dir,
                                                      target_size=(image_lengh,image_width),
                                                      batch_size=batch_size,
                                                      class_mode='categorical',
                                                      shuffle=True)
    cls_train = generator_train.classes
    cls_test = generator_test.class_indices
    print(cls_test,cls_train)
    return generator_train,generator_test

#构建网络
def creat_net(train_generator,validation_generator,batch_size,image_lengh,image_width):

    base_model = applications.resnet50.ResNet50(weights='imagenet', include_top=False, layers=tf.keras.layers,input_shape= (image_width,image_lengh,3))
    x = base_model.output
    x = GlobalAveragePooling2D(name='average_pool')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.0001), )(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization(name='bn_fc_01')(x)
    predictions = Dense(4, activation='softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    sgd = optimizers.Adam(lr=0.01, decay=1e-6)
    # Reduce=ReduceLROnPlateau(monitor='val_accuracy',
    #                          factor=0.1,
    #                          patience=2,
    #                          verbose=1,
    #                          mode='auto',
    #                          epsilon=0.0001,
    #                          cooldown=0,
    #                          min_lr=0)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #保存最优模型
    filepath = './模型/resnet_50_weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5'
    checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    model.fit_generator(train_generator, epochs=200, steps_per_epoch=1707//batch_size,validation_data=validation_generator,
                    validation_steps=264//batch_size, callbacks=[checkpoint])#Reduce])

    #绘制误差和准确率曲线
    loss = model.history.history['loss']
    val_loss = model.history.history['val_loss']
    epoches = range(1, len(loss) + 1)
    acc = model.history.history['accuracy']
    val_acc = model.history.history['val_accuracy']
    plt.subplot(121)
    plt.plot(epoches, loss, 'bo', label='training_loss')
    plt.plot(epoches, val_loss, 'r', label='validation_loss')
    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.title('losses of train and val')
    plt.legend()
    plt.subplot(122)
    plt.plot(epoches, acc, 'bo', label='training_acc')
    plt.plot(epoches, val_acc, 'r', label='validation_acc')
    plt.xlabel('epoches')
    plt.ylabel('acc')
    plt.title('accuracy of train and val')
    plt.legend()
    plt.show()

if __name__ =="__main__":
    #用于得到图片尺寸
    filename = '../垃圾分类/validation/plastic/plastic1.jpg'
    #训练集与测试集路径
    train_dir,test_dir = '../垃圾分类/train','../垃圾分类/validation'
    #得到图片尺寸
    image_lengh,image_width = get_image_size(filename)
    #生成图片
    generator_train,generator_test = image_process(batch_size=10,train_dir=train_dir,test_dir=test_dir,image_lengh=image_lengh,image_width=image_width)
    print("图片处理完成")
    #构建网络+训练
    creat_net(generator_train,generator_test,batch_size=10,image_lengh=image_lengh,image_width=image_width)
    print("结束")