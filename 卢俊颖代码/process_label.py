import os
import io
import numpy as np
import cv2

''' 异常图像 '''
outliers = [False for i in range(0, int(1e4))]   # 是否为异常图像

f = open('./dataset/outliers_too_dark.txt', 'r', encoding ='utf-8')

for line in f.readlines():
    lineArr = line.strip('\n')  # strip 去除首尾的指定字符；
    # outliers.append(lineArr)
    outliers[int(lineArr)] = True

f.close()

f = open('./dataset/outliers_too_bright.txt', 'r', encoding='utf-8')

for line in f.readlines():
    lineArr = line.strip('\n')  # strip 去除首尾的指定字符；
    # outliers.append(lineArr)
    outliers[int(lineArr)] = True

f.close()

'''
sex  # 性别
female: 0
male: 1

age  # 年龄
child: 0
teen: 1
adult: 2
senior: 3

race  # 种族
white: 0
asian: 1
black: 2

face  # 表情
smiling: 0
serious: 1
'''

''' 将faceDR文件中的文件名和标签写入txt文件中 '''
f = open('./dataset/faceDR', 'r', encoding ='utf-8')

with open('./dataset/label.txt', 'w') as t:
    for line in f.readlines():
        lineArr = line.strip(' ').split(' ')  # strip 去除首尾的指定字符；

        if len(lineArr) == 3:  # (_missing descriptor)
            pass
        elif outliers[int(lineArr[0])]:  # outliers
            pass
        else:
            sex = '0'   # 性别
            age = '0'   # 年龄
            race = '0'  # 种族
            face = '0'  # 表情
            if lineArr[3] == 'female)':  # 性别
                sex = '0'
            elif lineArr[3] == 'male)':
                sex = '1'
            if lineArr[6] == 'child)':  # 年龄
                age = '0'
            elif lineArr[6] == 'teen)':
                age = '1'
            elif lineArr[6] == 'adult)':
                age = '2'
            elif lineArr[6] == 'senior)':
                age = '3'
            if lineArr[8] == 'white)':  # 种族
                race = '0'
            elif lineArr[8] == 'asian)':
                race = '1'
            elif lineArr[8] == 'black)':
                race = '2'
            if lineArr[10] == 'smiling)':  # 表情
                face = '0'
            elif lineArr[10] == 'serious)':
                face = '1'
            t.write(lineArr[0] + ".jpg " + sex + " " + age + " " + race + " " + face + "\n")

f.close()


''' 将faceDS文件中的文件名和标签写入txt文件中 '''
f = open('./dataset/faceDS', 'r', encoding ='utf-8')

with open('./dataset/label.txt', 'a') as t:
    for line in f.readlines():
        lineArr = line.strip(' ').split(' ')  # strip 去除首尾的指定字符；

        if len(lineArr) == 3:  # (_missing descriptor)
            pass
        elif outliers[int(lineArr[0])]:  # outliers
            pass
        else:
            sex = '0'  # 性别
            age = '0'  # 年龄
            race = '0'  # 种族
            face = '0'  # 表情
            if lineArr[3] == 'female)':  # 性别
                sex = '0'
            elif lineArr[3] == 'male)':
                sex = '1'
            if lineArr[6] == 'child)':  # 年龄
                age = '0'
            elif lineArr[6] == 'teen)':
                age = '1'
            elif lineArr[6] == 'adult)':
                age = '2'
            elif lineArr[6] == 'senior)':
                age = '3'
            if lineArr[8] == 'white)':  # 种族
                race = '0'
            elif lineArr[8] == 'asian)':
                race = '1'
            elif lineArr[8] == 'black)':
                race = '2'
            if lineArr[10] == 'smiling)':  # 表情
                face = '0'
            elif lineArr[10] == 'serious)':
                face = '1'
            t.write(lineArr[0] + ".jpg " + sex + " " + age + " " + race + " " + face + "\n")

f.close()