# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import os
import cv2
import write2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing

data_dir = 'train'
label_dir = 'train.txt'
data_list = os.listdir(data_dir)

img_path = []  # 用于存放jpg文件的路径    3993
labels = []  # 标签                  3993
images = []

with open(label_dir, 'r') as f:
    for line in f.readlines():
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()  # words[0]代表xxxx.jpg; word[1]代表0或1

        path = os.path.join(data_dir, words[0])
        img_path.append(path)


def read_img(img_path):
    for i in range(len(img_path)):
        if os.path.isfile(img_path[i]):
            images.append(cv2.imread(img_path[i], -1).flatten())  #图片处理
read_img(img_path)



a=write2.labelprocess()
labels=np.array(a)


images = np.array(images)

print(images.shape)

#图片降维
pca = PCA(n_components=40)
pca = pca.fit(images)
images = pca.transform(images)
print(images.shape)
# 结果返回三列特征，也就是说3是最好的超参数取值

scaler = StandardScaler()
images = scaler.fit_transform(images)
print(images.shape)
#性别分类测试
class bayes_sexmodel():
    def __int__(self):
        pass
    def load_data(self):

        train_x, test_x, train_y, test_y = train_test_split(images,labels[0],test_size=0.3,random_state=123)
        return train_x, test_x, train_y, test_y
    def train_model(self, train_x, train_y):
        clf = GaussianNB()
        clf.fit(train_x, train_y)
        return clf
    def proba_data(self, clf, test_x, test_y):
        y_predict = clf.predict(test_x)
        y_proba = clf.predict_proba(test_x)
        accuracy = metrics.accuracy_score(test_y, y_predict) * 100

        print('性别测试的准确率是:',accuracy)
       # print('The result of predict is: \n', tot.head())
        return accuracy#, tot
    def exc_p(self):
        train_x, test_x, train_y, test_y = self.load_data()
        clf = self.train_model(train_x, train_y)
        res = self.proba_data(clf, test_x, test_y)
        return res
if __name__ == '__main__':
   bayes_sexmodel().exc_p()

class bayes_agemodel():
    def __int__(self):
        pass
    def load_data(self):

        train_x, test_x, train_y, test_y = train_test_split(images,labels[1],test_size=0.3,random_state=123)
        return train_x, test_x, train_y, test_y
    def train_model(self, train_x, train_y):
        clf = GaussianNB()
        clf.fit(train_x, train_y)
        return clf
    def proba_data(self, clf, test_x, test_y):
        y_predict = clf.predict(test_x)
        y_proba = clf.predict_proba(test_x)
        accuracy = metrics.accuracy_score(test_y, y_predict) * 100

        print('年龄测试的准确率是: ',accuracy)
       # print('The result of predict is: \n', tot.head())
        return accuracy#, tot
    def exc_p(self):
        train_x, test_x, train_y, test_y = self.load_data()
        clf = self.train_model(train_x, train_y)
        res = self.proba_data(clf, test_x, test_y)
        return res
if __name__ == '__main__':
   bayes_agemodel().exc_p()

class bayes_racemodel():
    def __int__(self):
        pass
    def load_data(self):

        train_x, test_x, train_y, test_y = train_test_split(images,labels[2],test_size=0.3,random_state=123)
        return train_x, test_x, train_y, test_y
    def train_model(self, train_x, train_y):
        clf = GaussianNB()
        clf.fit(train_x, train_y)
        return clf
    def proba_data(self, clf, test_x, test_y):
        y_predict = clf.predict(test_x)
        y_proba = clf.predict_proba(test_x)
        accuracy = metrics.accuracy_score(test_y, y_predict) * 100

        print('种族测试的准确率是: ',accuracy)
       # print('The result of predict is: \n', tot.head())
        return accuracy#, tot
    def exc_p(self):
        train_x, test_x, train_y, test_y = self.load_data()
        clf = self.train_model(train_x, train_y)
        res = self.proba_data(clf, test_x, test_y)
        return res
if __name__ == '__main__':
   bayes_racemodel().exc_p()

class bayes_facemodel():
    def __int__(self):
        pass
    def load_data(self):

        train_x, test_x, train_y, test_y = train_test_split(images,labels[3],test_size=0.3,random_state=123)
        return train_x, test_x, train_y, test_y
    def train_model(self, train_x, train_y):
        clf = GaussianNB()
        clf.fit(train_x, train_y)
        return clf
    def proba_data(self, clf, test_x, test_y):
        y_predict = clf.predict(test_x)
        y_proba = clf.predict_proba(test_x)
        accuracy = metrics.accuracy_score(test_y, y_predict) * 100

        print('表情测试的准确率是:',accuracy)
       # print('The result of predict is: \n', tot.head())
        return accuracy#, tot
    def exc_p(self):
        train_x, test_x, train_y, test_y = self.load_data()
        clf = self.train_model(train_x, train_y)
        res = self.proba_data(clf, test_x, test_y)
        return res
if __name__ == '__main__':
   bayes_facemodel().exc_p()