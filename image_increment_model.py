#-*- coding:utf-8 -*


import os
from os.path import join
from PIL import Image
import os
# import ConfigParser
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics, linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn import svm

import image_feature
import image_predict
from config import *
import configparser



#全局变量
# config = configparser.ConfigParser()
# config.read("./config.ini")
# model_path = config.get("global", "model_path") #模型存放路径


#训练模型
def trainModel(data, label):

    print("trainning process >>>>>>>>>>>>>>>>>>>>>>")

    # model = svm.SVC(decision_function_shape='ovo',kernel='rbf')
    # scores = cross_val_score(model, data, label, cv=10)
    # print("rbf: ",scores.mean())
    # model.fit(data, label)

    # 普通的
    # model = svm.SVC(decision_function_shape='ovo', kernel='linear')
    # scores = cross_val_score(model, data, label, cv=10)
    # print("linear: ", scores.mean())
    # model.fit(data, label)

    # model = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
    # scores = cross_val_score(model, data, label, cv=10)  # 交叉检验，计算模型平均准确率
    # print("rf: ", scores.mean())
    # model.fit(data, label)  # 拟合模型

    all = np.unique(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
    #增量学习
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        model.partial_fit(data, label, classes=all)  # 拟合模型
    else:
        model = linear_model.SGDClassifier()
        model.partial_fit(data, label, classes=all)


    predict = model.predict(data)
    acc = 0
    for num in range(len(label)):
        if predict[num] == label[num]:
            acc += 1
            # print("predict:", predict[num], "\tlabel: ", label[num])
    print("model acc: ", acc/len(label))

    # 模型持久化，保存到本地
    joblib.dump(model, model_path)
    print("model save success!")

    return model


#训练模型 #增量学习
def trainModel_increment(minibatch_train_iterators):
    print("trainning process >>>>>>>>>>>>>>>>>>>>>>")
    all = np.unique(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
    model = linear_model.SGDClassifier(loss="hinge", penalty="l2")
    for i, (data, label) in enumerate(minibatch_train_iterators):
        # 使用 partial_fit ，并在第一次调用 partial_fit 的时候指定 classes
        model.partial_fit(data, label, classes=all)
        print("{} time".format(i))  # 当前次数
        print("{} score".format(model.score(data, label)))  # 在测试集上看效果
    # 模型持久化，保存到本地
    joblib.dump(model, model_path)
    print("model save success!")
    return model

def trainModel_increment3():
    print("trainning process >>>>>>>>>>>>>>>>>>>>>>")
    all = np.unique(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
    model = linear_model.SGDClassifier(loss="hinge", penalty="l2")
    for i, (data, label) in enumerate(minibatch_train_iterators):
        # 使用 partial_fit ，并在第一次调用 partial_fit 的时候指定 classes
        model.partial_fit(data, label, classes=all)
        print("{} time".format(i))  # 当前次数
        print("{} score".format(model.score(data, label)))  # 在测试集上看效果
    # 模型持久化，保存到本地
    joblib.dump(model, model_path)
    print("model save success!")
    return model

def trainModel_increment2(data, label):
    print("trainning process >>>>>>>>>>>>>>>>>>>>>>")
    all = np.unique(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
    model = linear_model.SGDClassifier(loss="hinge", penalty="l2")
    # 使用 partial_fit ，并在第一次调用 partial_fit 的时候指定 classes
    model.partial_fit(data, label, classes=all)
    print("{} score".format(model.score(data, label)))  # 在测试集上看效果
    # 模型持久化，保存到本地
    joblib.dump(model, model_path)
    print("model save success!")
    return model




