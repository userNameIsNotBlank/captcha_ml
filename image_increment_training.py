#-*- coding:utf-8 -*
import os

import numpy as np
from PIL import Image

import image_increment_model
import image_process, image_feature, image_model
from config import *
import configparser


#读取配置文件
# config = configparser.ConfigParser()
# config.read("./config.ini")
# captcha_path = config.get("global", "captcha_path") #训练集验证码存放路径
# captcha__clean_path = config.get("global", "captcha__clean_path") #训练集验证码清理存放路径
# train_data_path = config.get("global", "train_data_path") #训练集存放路径
# model_path = config.get("global", "model_path") #模型存放路径
# test_data_path = config.get("global", "test_data_path") #测试集验证码存放路径
#
# image_character_num = config.get("global", "image_character_num") #识别的验证码个数
# threshold_grey = config.get("global", "threshold_grey") #图像粗处理的灰度阈值
# image_width = config.get("global", "image_width") #标准化的图像宽度（像素）
# image_height = config.get("global", "image_height") #标准化的图像高度（像素）




def iter_minibatches():
    '''
    迭代器
    给定文件流（比如一个大文件），每次输出minibatch_size行，默认选择1k行
    将输出转化成numpy输出，返回X, y
    '''
    image_array = []
    image_label = []
    feature = []
    max = 1000
    for label in os.listdir(train_data_path):  # 获取目录下的所有文件
        print(label)
        label_path = train_data_path + '/' + label
        for image_path in os.listdir(label_path):
            image = Image.open(label_path + '/' + image_path)
            image_array.append(image)
            image_label.append(label)
            if len(image_label) > max:
                for num, image in enumerate(image_array):
                    feature_vec = image_feature.feature_transfer(image)
                    feature.append(feature_vec)
                yield feature, image_label
                image_array = []
                image_label = []
                feature = []
    if len(image_label) > 0:
        for num, image in enumerate(image_array):
            feature_vec = image_feature.feature_transfer(image)
            feature.append(feature_vec)
        yield feature, image_label


if __name__ == '__main__':
    minibatch_train_iterators = iter_minibatches()
    image_increment_model.trainModel_increment(minibatch_train_iterators)



