
# coding: utf-8

# In[1]:
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
start_time = time.time()
import numpy as np
import pandas as pd
import keras
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, Flatten, Activation, Conv1D, BatchNormalization 
from keras.models import Model
from keras.layers.pooling import AveragePooling1D
from keras.optimizers import Adam,SGD,sgd
from keras.layers.wrappers import Bidirectional
from keras.utils.vis_utils import plot_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score,f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt

import tensorflow as tf
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu80%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.compat.v1.Session(config = config)

# In[3]:
data = pd.read_csv('.\equals_data_220000.csv', sep = ',', names = ['domain', 'index'])[0:]

# data = data.sample(frac=1) # 随机抽样
data=data.reset_index(drop=True)

domain = data['domain']
index = data['index']

domain_ = np.array(domain)
domain_ = np.array([[ord(j) for j in i] + [0 for k in range(53 - len(i))]
                    if len(i) <= 53 else [ord(i[j]) for j in range(53)]
                    for i in domain_])

index = np.array([[1 if i == 0 else 0, i] for i in index])

domain_train = domain_[44000:]
domain_test = domain_[:44000]
index_train = index[44000:]
index_test = index[:44000]

print(domain_.shape)
print(domain_train.shape)
print(domain_test.shape)

print(index.shape)
print(index_train.shape)
print(index_test.shape)

data.head()

# In[2]:

main_input = Input(shape=(53,), name='main_input')
x = Embedding(output_dim=128, input_dim=176000, input_length=53)(main_input)
print(x)
# x = keras.layers.core.Reshape((53,128,1))
# print("reshape后的X")
# print(x)

CNN3 = Conv1D(64, 3, padding='same', activation='relu')(x)
CNN3 = BatchNormalization()(CNN3)
CNN3 = AveragePooling1D(pool_size=14, strides=14, padding='same')(CNN3)
CNN3 = Flatten()(CNN3)
CNN3 = Dropout(0.3)(CNN3)

CNN4 = Conv1D(64, 4, padding='same', activation='relu')(x)
CNN4 = BatchNormalization()(CNN4)
CNN4 = AveragePooling1D(pool_size=14, strides=14, padding='same')(CNN4)
CNN4 = Flatten()(CNN4)
CNN4 = Dropout(0.3)(CNN4)

CNN5 = Conv1D(64, 5, padding='same', activation='relu')(x)
CNN5 = BatchNormalization()(CNN5)
CNN5 = AveragePooling1D(pool_size=14, strides=14, padding='same')(CNN5)
CNN5 = Flatten()(CNN5)
CNN5 = Dropout(0.3)(CNN5)

Bi_RNN = Bidirectional(LSTM(128))(x)
Bi_RNN = BatchNormalization()(Bi_RNN)
Bi_RNN = Dropout(0.3)(Bi_RNN)

x = keras.layers.concatenate([CNN3, CNN4, CNN5,Bi_RNN])
x = BatchNormalization()(x)

x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

main_output = Dense(2, activation='sigmoid', name='main_output')(x)
model = Model(inputs=[main_input], outputs=[main_output])

plot_model(model, to_file='my_model.png', show_shapes=True)

# In[8]:
'''
from sklearn.metrics import roc_auc_score
from keras import backend as K
import tensorflow as tf
# AUC for a binary classifier
def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N

# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P
'''
# In[9]:
#接着在模型的compile中设置metrics
adam = Adam(lr=0.002)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# In[ ]:
print('\nTraining ------------')
# Another way to train the model
model.fit(domain_train, index_train, epochs=20, batch_size=128)

# In[8]:
model.save('my_model.h5')

# In[9]:
print('\n\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(domain_test, index_test)
print('test loss: ', loss)
print('test accuracy: ', accuracy)

print("\nPredicting ------------")
index_pred = model.predict(domain_test)
index_pred = [np.argmax(y) for y in index_pred]  # 取出y中元素最大值所对应的索引
index_test = [np.argmax(y) for y in index_test]
# 二分类　ROC曲线
# roc_curve:真正率（True Positive Rate , TPR）或灵敏度（sensitivity）
# 横坐标：假正率（False Positive Rate , FPR）
fpr, tpr, thresholds_keras = roc_curve(index_test, index_pred)
auc = auc(fpr, tpr)
print("AUC : ", auc)
print("---costing %s seconds ---" % (time.time() - start_time))
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig("ROC_2分类.png")
plt.show()