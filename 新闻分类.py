from keras.datasets import reuters
import numpy as np
import matplotlib.pyplot as plt
from keras import models,layers

(train_data,train_labels),(test_data,test_labels) = reuters.load_data(num_words=10000)

#数据向量化
def vetorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results
x_train = vetorize_sequences(train_data)
x_test = vetorize_sequences(test_data)

#标签向量化
def to_one_hot(labels,dimension=46):
    results = np.zeros((len(labels),dimension))
    for i,label in enumerate(labels):
        results[i,label] = 1.
    return results
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

'''keras内置方法实现标签向量化
from keras.utils.np_utils import to_categorical
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)
'''

#构建网络
models = models.Sequential()
models.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
models.add(layers.Dense(64,activation='relu'))
models.add(layers.Dense(46,activation='softmax'))

models.compile(optimizer='rmsprop',
               loss='categorical_crossentropy',
                metrics=['acc']
               )

#数据集分类
x_val = x_train[0:1000]
pratial_x_train = x_train[1000:]

y_val = one_hot_train_labels[0:1000]
pratial_y_train = one_hot_train_labels[1000:]

#开始训练
history = models.fit(pratial_x_train,
           pratial_y_train,
           epochs=20,
           batch_size=512,
           validation_data=(x_val,y_val)
)
'''
history = models.fit(pratial_x_train,
           pratial_y_train,
           epochs=9,
           batch_size=512,
           validation_data=(x_val,y_val)
results = model.evaluate(x_test,one_hot_test_label)
'''
#绘制误差和准确度制曲线
loss = history.history['loss']
val_loss = history.history['val_loss']
epoches = range(1,len(loss)+1)

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.subplot(121)
plt.plot(epoches,loss,'bo',label = 'training_loss')
plt.plot(epoches,val_loss,'r',label = 'validation_loss')
plt.xlabel('epoches')
plt.ylabel('loss')
plt.title('losses of train and val')
plt.legend()

plt.subplot(122)
plt.plot(epoches,acc,'bo',label = 'training_acc')
plt.plot(epoches,val_acc,'r',label = 'validation_acc')
plt.xlabel('epoches')
plt.ylabel('acc')
plt.title('accuracy of train and val')
plt.legend()

plt.show()

#
import copy
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
print(float(np.sum(hits_array))/len(test_labels))




