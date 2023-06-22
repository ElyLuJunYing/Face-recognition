from sklearn import preprocessing
import os
import io
import numpy as np
import cv2
f = open( 'faceDR', 'r',encoding = 'utf-8' )
g = open( 'faceDS', 'r',encoding = 'utf-8' )
a = []
#with open('./DR.txt', 'w') as t:
def labelprocess():

    for line in f.readlines():
        lineArr = line.strip(' ').split(' ')  # strip 去除首尾的指定字符；

        if len(lineArr) == 3:
            pass
        else:
            lineArr.remove("(_sex")
            lineArr.remove("(_age")
            lineArr.remove("(_race")
            lineArr.remove("(_face")
            del lineArr[1]
            del lineArr[2]
            del lineArr[5:]
            del lineArr[0]
            data = lineArr
            if data[1]=='chil)':
                data[1]='child)'
            elif data[1]=='adulte)':
                data[1]='adult)'
            elif data[2]=='whit)':
                data[2]='white)'
            elif data[2]=='whitee)':
                data[2]='white)'
            elif data[3]=='smilin)':
                data[3]='smiling)'
            elif data[3]=='erious)':
                data[3]='serious)'

            enc = preprocessing.LabelEncoder()
            data = enc.fit_transform(data)
            data=list(data)
            a.append(data)
    #b=np.array(a)
    #print(b.shape)
    for line in g.readlines():
        lineArr = line.strip(' ').split(' ')  # strip 去除首尾的指定字符；

        if len(lineArr) == 3:
            pass
        else:
            lineArr.remove("(_sex")
            lineArr.remove("(_age")
            lineArr.remove("(_race")
            lineArr.remove("(_face")
            del lineArr[1]
            del lineArr[2]
            del lineArr[5:]
            del lineArr[0]
            data = lineArr
            enc = preprocessing.LabelEncoder()
            data = enc.fit_transform(data)
            data = list(data)
            a.append(data)


    return a
a=labelprocess()
c=np.array(a)
print(c)

#labels=np.array(a)
#labels=labels[:,0]
#print(labels.shape)

#print(labels.shape)
#print(labels)
f.close
g.close
