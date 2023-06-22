import os
import numpy as np
import cv2
import write
import write2
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import preprocessing

data_dir = 'train'      #放有提取出的图片的文件
label_dir = 'train.txt' #放有处理后的标签地址的txt文件
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



#a=write2.labelprocess()   #调用标签处理的数据
a=write.labelprocess() #调用错误处理方法标签处理的数据
labels=np.array(a)       #标签数据列表转化为数组


images = np.array(images) #图片数据列表转化为数组
print(images.shape)

#图片降维
pca = PCA(n_components=40)
pca = pca.fit(images)
images = pca.transform(images) #降维至40
print(images.shape)


scaler = StandardScaler()
images = scaler.fit_transform(images) #标准化
print(images.shape)

#性别分类测试
imagestrain,imagestest,labelstrain,labelstest=train_test_split(images,labels[:,0],test_size=0.3,random_state=42)

knn_sex=KNeighborsClassifier(n_neighbors=4,p=1,weights='distance').fit(imagestrain,labelstrain) #使用KNN分类器进行数据训练
print("性别分类得分为：",knn_sex.score(imagestest,labelstest))                 #输出性别分类分数指标



#report=classification_report(knn_sex.predict(imagestest),labelstest) #性别分类模型评估报告输出
#print(report)
#年龄分类测试
imagestrain,imagestest,labelstrain,labelstest=train_test_split(images,labels[:,1],test_size=0.3,random_state=42)

knn_age=KNeighborsClassifier(n_neighbors=4,p=1,weights='distance').fit(imagestrain,labelstrain) #使用KNN分类器进行数据训练
print("年龄分类得分为：",knn_age.score(imagestest,labelstest))                #输出年龄分类分数指标

#report=classification_report(knn_age.predict(imagestest),labelstest) #年龄分类模型评估报告输出
#print(report)
#肤色分类测试
imagestrain,imagestest,labelstrain,labelstest=train_test_split(images,labels[:,2],test_size=0.3,random_state=42)

knn_race=KNeighborsClassifier(n_neighbors=4,p=1,weights='distance').fit(imagestrain,labelstrain)#使用KNN分类器进行数据训练
print("肤色分类得分为：",knn_race.score(imagestest,labelstest))           #输出肤色分类分数指标

#report=classification_report(knn_race.predict(imagestest),labelstest) #肤色分类模型评估报告输出
#print(report)
#表情分类测试
imagestrain,imagestest,labelstrain,labelstest=train_test_split(images,labels[:,3],test_size=0.3,random_state=42)

knn_face=KNeighborsClassifier(n_neighbors=4,p=1,weights='distance').fit(imagestrain,labelstrain)#使用KNN分类器进行数据训练
print("表情分类得分为：",knn_face.score(imagestest,labelstest))

#report=classification_report(knn_face.predict(imagestest),labelstest) #表情分类模型评估报告输出
#print(report)

param_grid = [
    {
        'weights': ['uniform'], # 参数取值范围
        'n_neighbors': [i for i in range(3, 8)]  # 使用其他方式如np.arange()也可以
        # 这里没有p参数
    },
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(3, 8)],
        'p': [i for i in range(1, 2)]
    }
]
knn_clf = KNeighborsClassifier()  # 默认参数，创建空分类器
grid_search = GridSearchCV(knn_clf, param_grid, cv=5)  # 网格搜参
grid_search.fit(imagestrain, labelstrain)  # 网格搜索训练模型，比较耗时，约4分钟

print(grid_search.best_params_)



