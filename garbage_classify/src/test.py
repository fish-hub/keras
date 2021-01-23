import os
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_and_predict(imgs_path,model_path):
    model = load_model(model_path)
    imgs = os.listdir(imgs_path)
    i = 0
    target =''
    #画布创建
    plt.figure(figsize=(10,20))
    for img in imgs:
        #得到真实标签
        for s in img:
            try:
                x = int(s)
                break
            except:
                target += s
        #显示图片(inceptionv3模型由于修改了图片尺寸所以需要把下面两个注释打开)
        img_path = imgs_path+'/'+img
        image1 = cv2.imread(img_path)
        #image1 = cv2.resize(image1, (229, 229))
        plt.subplot(4, 4, i+1)
        plt.imshow(image1)
        #进行预测
        img = image.load_img(img_path)
        img = image.img_to_array(img)
        img = 1.0/255 * img
        #img = cv2.resize(img, (229, 229))
        img = np.expand_dims(img, axis=0)
        y = model.predict(img)
        labels = {0: 'glass', 1: 'metal', 2: 'paper', 3: 'plastic'}
        y_predict = labels[np.argmax(y)]
        #图片title
        plt.title('pred:%s / truth:%s' % (y_predict, target))
        i=i+1
        target = ''

if __name__ == '__main__':
    #验证集路径
    imgs_path = "../test_picture"
    #模型路径
    model_path = './模型/ResNet50V2_weights-improvement-25-0.95.hdf5'
    #预测
    load_and_predict(imgs_path,model_path)
    plt.show()
    print("结束")