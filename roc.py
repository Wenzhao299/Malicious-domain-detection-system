from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
start_time = time.time()
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score,f1_score
from keras.optimizers import Adam,SGD,sgd
from keras.models import load_model

data = pd.read_csv('E:\\曹文钊\\大创\\代码\\equals_data_220000.csv', sep = ',', names = ['domain', 'index'])[0:]

# data = data.sample(frac=1) # 随机抽样
data=data.reset_index(drop=True)

domain = data['domain']
index = data['index']

domain_ = np.array(domain)
domain_ = np.array([[ord(j) for j in i] + [0 for k in range(53 - len(i))]
                    if len(i) <= 53 else [ord(i[j]) for j in range(53)]
                    for i in domain_])

index = np.array([[1 if i == 0 else 0, i] for i in index])

X_train = domain_[44000:]
X_valid = domain_[:44000]
Y_train = index[44000:]
Y_valid = index[:44000]


print('获取模型')
model = load_model('E:\\曹文钊\\大创\\代码\\my_model.h5')
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy')


print("Predicting")
Y_pred = model.predict(X_valid)
Y_pred = [np.argmax(y) for y in Y_pred]  # 取出y中元素最大值所对应的索引
Y_valid = [np.argmax(y) for y in Y_valid]



# # micro：多分类　　
# # weighted：不均衡数量的类来说，计算二分类metrics的平均
# # macro：计算二分类metrics的均值，为每个类给出相同权重的分值。
# precision = precision_score(Y_valid, Y_pred, average='weighted')
# recall = recall_score(Y_valid, Y_pred, average='weighted')
# f1_score = f1_score(Y_valid, Y_pred, average='weighted')
# accuracy_score = accuracy_score(Y_valid, Y_pred)
# print("Precision_score:",precision)
# print("Recall_score:",recall)
# print("F1_score:",f1_score)
# print("Accuracy_score:",accuracy_score)



# 二分类　ＲＯＣ曲线
# roc_curve:真正率（True Positive Rate , TPR）或灵敏度（sensitivity）
# 横坐标：假正率（False Positive Rate , FPR）
fpr, tpr, thresholds_keras = roc_curve(Y_valid, Y_pred)
auc = auc(fpr, tpr)
print("AUC : ", auc)
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig("ROC_2分类.png")
plt.show()


print("--- %s seconds ---" % (time.time() - start_time))