import os
import io
import numpy as np
import cv2
import shutil
import random

data_dir = 'D:\\pycharm\\face\\data\\train'
label_dir = 'D:\\pycharm\\face\\data\\face.txt'


new_data_2 = r'D:\pycharm\face2\face\train\2'
new_data_1 = r'D:\pycharm\face2\face\train\1'
new_data_0 = r'D:\pycharm\face2\face\train\0'

data2=[]
data1=[]
data0=[]


test_2 = r'D:\pycharm\face2\face\test\2'
test_1 = r'D:\pycharm\face2\face\test\1'
test_0 = r'D:\pycharm\face2\face\test\0'

# with open(label_dir, 'r') as f:
#     for line in f.readlines():
#         line = line.strip('\n')
#         line = line.rstrip()
#         words = line.split()         # words[0]代表xxxx.jpg; word[1]代表0或1
#
#         path = os.path.join(data_dir,words[0])
#         if words[1] == '0':
#             shutil.copy(path,r'D:\pycharm\face2\face\train\0')
#         if words[1] == '1':
#             shutil.copy(path,r'D:\pycharm\face2\face\train\1')
#         if words[1] == '2':
#             shutil.copy(path,r'D:\pycharm\face2\face\train\2')
#
#         # labels.append(words[1])      # 写入标签
# f.close()
############################################################

for filename in os.listdir(new_data_2):
    data2.append(os.path.join(new_data_2,filename))
for filename in os.listdir(new_data_1):
    data1.append(os.path.join(new_data_1,filename))
for filename in os.listdir(new_data_0):
    data0.append(os.path.join(new_data_0,filename))


test_2_file = random.sample(data2, round(len(data2)*0.2))
test_1_file = random.sample(data1, round(len(data1)*0.2))
test_0_file = random.sample(data0, round(len(data0)*0.2))

for path0 in test_0_file:
    shutil.move(path0,test_0)
for path1 in test_1_file:
    shutil.move(path1,test_1)
for path2 in test_2_file:
    shutil.move(path2,test_2)
