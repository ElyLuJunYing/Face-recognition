from sklearn import preprocessing
import os
import io
import numpy as np
import cv2
f = open( 'faceDR', 'r',encoding = 'utf-8' )
g = open( 'faceDS', 'r',encoding = 'utf-8' )
a = []

def labelprocess():

    for line in f.readlines():

        lineArr = line.strip(' ').split(' ')  # strip 去除首尾的指定字符；
        if len(lineArr) == 3:  #去除标签中不含信息的标签
            pass
       # if line[]
        else:
            lineArr.remove("(_sex")
            lineArr.remove("(_age")      #去除标签中的类别名称
            lineArr.remove("(_race")
            lineArr.remove("(_face")
            del lineArr[1]
            del lineArr[2]               #去除余下非数据的部分
            del lineArr[5:]
            del lineArr[0]
            data = lineArr

            data=list(data)
            if data[1]=='chil)':
                data[1]='child)'
            elif data[1]=='adulte)':
                data[1]='adult)'
            elif data[2]=='whit)':
                data[2]='white)'
            elif data[2]=='whitee)':    #将数据中错误的数据进行修正
                data[2]='white)'
            elif data[3]=='smilin)':
                data[3]='smiling)'
            elif data[3]=='erious)':
                data[3]='serious)'
            a.append(data)
   # b=np.array(a)
    #print(b.shape)
    for line in g.readlines():
        lineArr = line.strip(' ').split(' ')  # strip 去除首尾的指定字符；

        if len(lineArr) == 3:     #去除标签中的类别名称
            pass
        else:
            lineArr.remove("(_sex")
            lineArr.remove("(_age")   #去除标签中的类别名称
            lineArr.remove("(_race")
            lineArr.remove("(_face")
            del lineArr[1]
            del lineArr[2]
            del lineArr[5:]
            del lineArr[0]           #去除余下非数据的部分
            data = lineArr
            data = list(data)
            a.append(data)
    c=np.array(a)

    print(c)
    enc = preprocessing.LabelEncoder()
    d = []
    d.append(list(enc.fit_transform(c[:,0])))    #将2维数组进行标签编码，并按列提取存放在列表d中
    d.append(list(enc.fit_transform(c[:,1])))
    d.append(list(enc.fit_transform(c[:,2])))
    d.append(list(enc.fit_transform(c[:,3])))
    return d

d=labelprocess()
d=np.array(d)
print(d[2])
#labels=np.array(a)
#labels=labels[:,0]
#print(labels.shape)

#print(labels.shape)
#print(labels)
f.close
g.close