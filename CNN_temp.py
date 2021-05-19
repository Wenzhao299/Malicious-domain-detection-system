import numpy as np
import pandas as pd
import keras
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, Flatten, Activation, Conv1D, BatchNormalization, ReLU 
from keras.models import Model
from keras.layers.pooling import AveragePooling1D
from keras.optimizers import Adam
from keras.layers.wrappers import Bidirectional
from keras.utils.vis_utils import plot_model
# from keras.callbacks import TensorBoard


main_input = Input(shape=(53,), name='main_input')
x = Embedding(output_dim=128, input_dim=176000, input_length=53)(main_input)
print(x)
# x = keras.layers.core.Reshape((53,128,1))
# print("reshape后的X")
# print(x)


# CNN2 = Conv1D(64, 5, padding='same', activation='relu')(x)
# CNN2 = BatchNormalization()(CNN2)
# CNN2 = AveragePooling1D(pool_size=20, strides=16, padding='same')(CNN2)
# CNN2 = Flatten()(CNN2)
# CNN2 = Dropout(0.35)(CNN2)

CNN3 = Conv1D(64, 3, padding='same')(x)
CNN3 = BatchNormalization()(CNN3)
CNN3 = ReLU()(CNN3)
CNN3 = AveragePooling1D(pool_size=3, strides=3, padding='same')(CNN3)
CNN3 = Conv1D(64, 3, padding='same')(CNN3)
CNN3 = BatchNormalization()(CNN3)
CNN3 = ReLU()(CNN3)
CNN3 = AveragePooling1D(pool_size=3, strides=2, padding='same')(CNN3)
CNN3 = Conv1D(64, 3, padding='same')(CNN3)
CNN3 = BatchNormalization()(CNN3)
CNN3 = ReLU()(CNN3)
CNN3 = AveragePooling1D(pool_size=2, strides=2, padding='same')(CNN3)
CNN3 = Flatten()(CNN3)
CNN3 = Dropout(0.5)(CNN3)


CNN4 = Conv1D(64, 4, padding='same')(x)
CNN4 = BatchNormalization()(CNN4)
CNN4 = ReLU()(CNN4)
CNN4 = AveragePooling1D(pool_size=3, strides=3, padding='same')(CNN4)
CNN4 = Conv1D(64, 4, padding='same')(CNN4)
CNN4 = BatchNormalization()(CNN4)
CNN4 = ReLU()(CNN4)
CNN4 = AveragePooling1D(pool_size=3, strides=2, padding='same')(CNN4)
CNN4 = Conv1D(64, 4, padding='same')(CNN4)
CNN4 = BatchNormalization()(CNN4)
CNN4 = ReLU()(CNN4)
CNN4 = AveragePooling1D(pool_size=2, strides=2, padding='same')(CNN4)
CNN4 = Flatten()(CNN4)
CNN4 = Dropout(0.5)(CNN4)

CNN5 = Conv1D(64, 5, padding='same')(x)
CNN5 = BatchNormalization()(CNN5)
CNN5 = ReLU()(CNN5)
CNN5 = AveragePooling1D(pool_size=3, strides=3, padding='same')(CNN5)
CNN5 = Conv1D(64, 5, padding='same')(CNN5)
CNN5 = BatchNormalization()(CNN5)
CNN5 = ReLU()(CNN5)
CNN5 = AveragePooling1D(pool_size=3, strides=2, padding='same')(CNN5)
CNN5 = Conv1D(64, 5, padding='same')(CNN5)
CNN5 = BatchNormalization()(CNN5)
CNN5 = ReLU()(CNN5)
CNN5 = AveragePooling1D(pool_size=2, strides=2, padding='same')(CNN5)
CNN5 = Flatten()(CNN5)
CNN5 = Dropout(0.5)(CNN5)

Bi_RNN = Bidirectional(LSTM(128))(x)
Bi_RNN = Dropout(0.5)(Bi_RNN)
Bi_RNN = BatchNormalization()(Bi_RNN)

# x = keras.layers.concatenate([CNN2, CNN3, CNN4, CNN5])
x = keras.layers.concatenate([CNN3, CNN4, CNN5,Bi_RNN])
x = BatchNormalization()(x)

x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.35)(x)

x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.35)(x)

x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.35)(x)

main_output = Dense(2, activation='sigmoid', name='main_output')(x)
model = Model(inputs=[main_input], outputs=[main_output])

plot_model(model, to_file='model2.png', show_shapes=True)


data = pd.read_csv('./Data/deal_by_tiantian/equals_data_220000.csv', sep = ',', names = ['domain', 'index'])[0:]

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
print
print(domain_.shape)
print(domain_train.shape)
print(domain_test.shape)

print(index.shape)
print(index_train.shape)
print(index_test.shape)

data.head()


adam = Adam(lr=0.001)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


print('Training ------------')
# Another way to train the model
model.fit(domain_train, index_train, epochs=3, batch_size=128)

model.save('model_三层')

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(domain_test, index_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)