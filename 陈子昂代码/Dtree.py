import os
import numpy as np
import cv2
import write
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
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



a=write.labelprocess()
labels=np.array(a)

images = np.array(images)
print(images.shape)

#图片降维
pca = PCA(n_components=150)
pca = pca.fit(images)
images = pca.transform(images)
print(images.shape)
# 结果返回三列特征，也就是说3是最好的超参数取值

scaler = StandardScaler()
images = scaler.fit_transform(images)
print(images.shape)

#性别分类测试
imagestrain,imagestest,labelstrain,labelstest=train_test_split(images,labels[:,0],test_size=0.3,random_state=42)

dt_sex=DecisionTreeClassifier(criterion='gini', max_depth=30, min_impurity_decrease=0.1,min_samples_leaf=2).fit(imagestrain,labelstrain)
print("性别分类得分为：",dt_sex.score(imagestest,labelstest))

#年龄分类测试
imagestrain,imagestest,labelstrain,labelstest=train_test_split(images,labels[:,1],test_size=0.3,random_state=42)

dt_age=DecisionTreeClassifier(criterion='gini', max_depth=30, min_impurity_decrease=0.1,min_samples_leaf=2).fit(imagestrain,labelstrain)
print("年龄分类得分为：",dt_age.score(imagestest,labelstest))
#肤色分类测试
imagestrain,imagestest,labelstrain,labelstest=train_test_split(images,labels[:,2],test_size=0.3,random_state=42)

dt_race=DecisionTreeClassifier(criterion='gini', max_depth=30, min_impurity_decrease=0.1,min_samples_leaf=2).fit(imagestrain,labelstrain)
print("肤色分类得分为：",dt_race.score(imagestest,labelstest))

#表情分类测试
imagestrain,imagestest,labelstrain,labelstest=train_test_split(images,labels[:,3],test_size=0.3,random_state=42)

dt_face=DecisionTreeClassifier(criterion='gini', max_depth=30, min_impurity_decrease=0.1,min_samples_leaf=2).fit(imagestrain,labelstrain)
print("表情分类得分为：",dt_face.score(imagestest,labelstest))

param_grid  = [{'criterion':['gini'],
                'max_depth':[30,50,60,100],
                'min_samples_leaf':[2,3,5,10],
                'min_impurity_decrease':[0.1,0.2,0.5]},
         {'max_depth': [30,60,100], 'min_impurity_decrease':[0.1,0.2,0.5]}]

dt_clf = DecisionTreeClassifier()  # 默认参数，创建空分类器
grid_search = GridSearchCV(dt_clf, param_grid,cv=5)  # 网格搜参
grid_search.fit(imagestrain, labelstrain)  # 网格搜索训练模型，比较耗时，约4分钟

print(grid_search.best_params_)
