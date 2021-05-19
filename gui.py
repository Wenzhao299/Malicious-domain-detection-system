import numpy as np
import pandas as pd
import keras
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, Flatten, Conv1D, BatchNormalization
from keras.models import Model, load_model
from keras.layers.pooling import AveragePooling1D
from keras.optimizers import Adam
from keras.layers.wrappers import Bidirectional

from tkinter import *

domain_train = 0
domain_test = 0
index_train = 0
index_test = 0
model = 0


def buildModel():
    global model
    main_input = Input(shape=(53,), name='main_input')
    x = Embedding(output_dim=128, input_dim=176000, input_length=53)(main_input)
    print(x)

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

    x = keras.layers.concatenate([CNN3, CNN4, CNN5, Bi_RNN])
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

    adam = Adam(lr=0.002)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


def loadDataset():
    data = pd.read_csv(open(entryDatasetPath.get()), sep=',', names=['domain', 'index'])[0:]
    data = data.reset_index(drop=True)

    domain = data['domain']
    index = data['index']

    domain_ = np.array(domain)
    domain_ = np.array([[ord(j) for j in i] + [0 for k in range(53 - len(i))]
                        if len(i) <= 53 else [ord(i[j]) for j in range(53)]
                        for i in domain_])

    index = np.array([[1 if i == 0 else 0, i] for i in index])

    global domain_train
    global domain_test
    global index_train
    global index_test
    domain_train = domain_[44000:]
    domain_test = domain_[:44000]
    index_train = index[44000:]
    index_test = index[:44000]

    print('数据集载入成功\n')

    print('输入数据shape:' + str(domain_.shape))
    print('训练数据shape:' + str(domain_train.shape))
    print('测试数据shape:' + str(domain_test.shape))

    print('输入标签shape:' + str(index.shape))
    print('训练标签shape:' + str(index_train.shape))
    print('测试标签shape:' + str(index_test.shape))


def train():
    buildModel()

    global domain_train
    global domain_test
    global index_train
    global index_test

    model.fit(domain_train, index_train, epochs=int(entryEpochs.get()), batch_size=128)

    model.save(entryModelPath.get())

    loss, accuracy = model.evaluate(domain_test, index_test)

    print('训练完成\n')
    print('测试数据集 loss: ' + str(loss))
    print('测试数据集 accuracy: ' + str(accuracy))


def loadModel():
    global model
    model = load_model(entryModel.get())
    print('模型载入成功\n')


def judge():
    domain = entryDomain.get()
    domain_data = np.array([([ord(j) for j in domain] + [0 for k in range(53 - len(domain))])
                      if len(domain) <= 53 else [ord(domain[j]) for j in range(53)]])
    pred = model.predict(domain_data)
    if pred[0][0] > pred[0][1]:
        print(domain + '：不是DGA\n')
    else:
        print(domain + '：是DGA\n')


def print(string):
    outputText.config(state=NORMAL)
    outputText.insert('insert', str(string)+'\n')
    outputText.config(state=DISABLED)
    outputText.see(END)


if __name__ == '__main__':
    root = Tk()
    root.title('DGA域名识别')
    width, height = 400, 500
    root.maxsize(width, height)
    root.minsize(width, height)

    trainFrame = LabelFrame(root, width=400, height=150, text='训练模型')\
        .grid(row=0, column=0, rowspan=4, columnspan=3)
    Label(root, text='输入数据集路径').grid(row=1, column=0)
    Label(root, text='输入模型保存位置').grid(row=2, column=0)
    Label(root, text='训练轮次').grid(row=3, column=0)
    entryDatasetPath = Entry(root)
    entryDatasetPath.grid(row=1, column=1)
    entryModelPath = Entry(root)
    entryModelPath.grid(row=2, column=1)
    entryEpochs = Entry(root)
    entryEpochs.grid(row=3, column=1)
    Button(root, text='载入数据集', command=loadDataset).grid(row=1, column=2)
    Button(root, text='开始训练', command=train).grid(row=2, column=2, rowspan=2)

    judgeFrame = LabelFrame(root, width=400, height=100, text='载入模型并检测')\
        .grid(row=4, column=0, rowspan=3, columnspan=3)
    Label(root, text='输入模型保存位置').grid(row=5, column=0)
    Label(root, text='输入要检测的域名').grid(row=6, column=0)
    entryModel = Entry(root)
    entryModel.grid(row=5, column=1)
    entryDomain = Entry(root)
    entryDomain.grid(row=6, column=1)
    Button(root, text='载入模型', command=loadModel).grid(row=5, column=2)
    Button(root, text='检测', command=judge).grid(row=6, column=2)

    outputFrame = LabelFrame(root, width=400, height=250, text='输出')\
        .grid(row=7, column=0, rowspan=3, columnspan=3)
    outputText = Text(root, width=40, height=10, state='disabled')
    outputText.grid(row=8, column=0, columnspan=3)

    root.mainloop()