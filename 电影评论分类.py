from keras.datasets import imdb
import numpy as np
from keras import models,layers,optimizers
import  matplotlib.pyplot as plt


#加载IMDB数据集
(train_data,train_labels),(test_data,test_labels) =imdb.load_data(path=
                        "/home/dzw/.keras/datasets/imdb.npz",num_words=10000)

#查看最大单词索引
#print(max([max(sequense) for sequense in train_data]))

'''
#将评论解码为汉字
word_index = imdb.get_word_index()
reverse_word_index = dict([(value,key) for (key,value)in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])
#print(decoded_review)
'''

#准备数据
#将整数序列编码为2进制矩阵 one-hot
def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#构建网络
model = models.Sequential()
model.add(layers.Dense(16,activation = 'relu',input_shape = (10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#编译模型，配置损失函数和优化器
model.compile(optimizer = 'rmsprop',
                loss='binary_crossentropy',
                metrics=['acc'])

'''
#自定义损失函数和优化器
model.compile(optimizer = optimizers.RMSprop,
              loss = losses.binary_crossentropy,
              metrics = [metrics.binary_accuracy])
'''

#留出验证集(分离训练集和验证集)
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#训练模型
'''
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=[x_val,y_val]
                    )
'''
#按照loss和accuracy曲线修改训练模型防止过拟合：

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=4,
                    batch_size=512,
                    validation_data=[x_val,y_val]
                    )



"""
#查看history对象中的history（字典），字典key包括val_acc,acc,val_loss,loss
history_dict=history.history
print(history_dict.keys())
"""

#绘制训练损失和验证损失图
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
print("sssssssssssssssssssssssssssss")

print(len(loss_values))
epochs = range(1,len(loss_values)+1)

#bo表示蓝色圆点
plt.subplot(121)
plt.plot(epochs,loss_values,'bo',label = 'training loss')
plt.plot(epochs,val_loss_values,'b',label = 'validation loss')
plt.title('training and validation loss')
plt.xlabel('epoches')
plt.ylabel('loss')
plt.legend()

plt.subplot(122)
acc = history_dict['acc']
val_acc = history_dict['val_acc']
plt.plot(epochs,acc,'bo',label = 'training acc')
plt.plot(epochs,val_acc,'b',label = 'validation acc')
plt.title('training and validation accuracy')
plt.xlabel('epoches')
plt.ylabel('accuracy')
plt.legend()
plt.show()

#模型用于预测
print(model.predict(x_test))