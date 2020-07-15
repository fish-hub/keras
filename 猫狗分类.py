from keras import layers,models,optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#1.数据预处理
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255) #把图像进行缩放

#1.1训练数据和测试数据生成器
train_dir = '/home/dzw/.keras/datasets/cats_and_dogs_small/train'
validation_dir = '/home/dzw/.keras/datasets/cats_and_dogs_small/validation'
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary'
)
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size = 20,
    class_mode = 'binary'
)

'''
#查看生成器返回数据形状
for data_batch,label_batch in train_generator:
    print('data batch shape:',data_batch.shape)
    print('label batch shape:',label_batch.shape)
'''

#定义网络结构
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())         #展平所有神经元
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#模型解释，使用二元交叉緔，优化器选择rmsprop（反向传播）,衡量指标选择准确率
model.compile(loss = 'binary_crossentropy',
              optimizer = optimizers.RMSprop(lr=1e-4),
              metrics = ['acc'])

#模型拟合
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,#一个epoches抽取100个批量，运行一百次梯度下降进入下一个epoch
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

#保存模型
model.save('cats_and_dogs_small_1.h5')
#绘制图像
fig,axes = plt.subplots(nrows=1,ncols=2)
##训练集准确率和损失
acc = history.history['acc']
loss = history.history['loss']
##在验证集上的误差和损失
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']
epoches = range(1,len(acc)+1)

axes[0].plot(epoches,acc,'bo',label = 'training_acc')
axes[0].plot(epoches,val_acc,'b',label = 'validation_acc')
axes[0].set_title('training and validation acc')
axes[0].legend()
axes[1].plot(epoches,loss,'bo',label = 'training_loss')
axes[1].plot(epoches,val_loss,'b',label = 'validation_loss')
axes[1].set_title('training and validation loss')
axes[1].legend()
plt.show()