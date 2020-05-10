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

    model = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
    scores = cross_val_score(model, data, label, cv=10)  # 交叉检验，计算模型平均准确率
    print("rf: ", scores.mean())
    model.fit(data, label)  # 拟合模型

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


#测试模型
def testModel(data, label):
    #读取模型
    model = joblib.load(model_path)
    #预测
    predict_list = model.predict(data)
    #print classification_report(label, predict_list)#按类别分类的各种指标
    print("\ntest process >>>>>>>>>>>>>>>>>>>>>>>>")
    print("test precision: ",metrics.precision_score(label, predict_list))#precision
    print("test recall: ",metrics.recall_score(label, predict_list))#recall
    print("test f1 score: ",metrics.f1_score(label, predict_list))#f1 score
    print("confusion matrix:")
    print(confusion_matrix(label, predict_list))#混淆矩阵



