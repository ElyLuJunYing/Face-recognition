import os
import io
import numpy as np
import cv2

# 以下代码用于将faceDS文件中的文件名和标签写入txt文件中

f = open( 'D:\\pycharm\\face\\faceDS', 'r',encoding = 'utf-8' )
images_dir = r'D:\pycharm\face\data\train'

with open('./face.txt', 'w') as t:
    for line in f.readlines():
        lineArr = line.strip(' ').split(' ')  # strip 去除首尾的指定字符；
        # lineArr = lineArr.remove('(_sex')



        if os.path.isfile(os.path.join(images_dir,lineArr[0]) + '.jpg'):
            # num = np.shape(lineArr)[0]
            if len(lineArr) == 3:
                pass
            else:
                lineArr = lineArr  # [0]代表文件名，[1]代表(_sex，[2]空，[3]代表female)或者male)
                # print(lineArr)
                if lineArr[10] == 'serious)':
                    t.write(lineArr[0] + ".jpg " + '0' + "\n")
                if lineArr[10] == 'smiling)':
                    t.write(lineArr[0] + ".jpg " + '1' + "\n")
                if lineArr[10] == 'funny)':
                    t.write(lineArr[0] + ".jpg " + '2' + "\n")



            # else:
            #     print(lineArr)
t.close()


# with open('./race.txt', 'w') as t:
#     for line in f.readlines():
#         lineArr = line.strip(' ').split(' ')  # strip 去除首尾的指定字符；
#         # lineArr = lineArr.remove('(_sex')
#
#         # num = np.shape(lineArr)[0]
#         if os.path.isfile(os.path.join(images_dir, lineArr[0]) + '.jpg'):
#             if len(lineArr) == 3:
#                 pass
#             else:
#                 lineArr = lineArr  # [0]代表文件名，[1]代表(_sex，[2]空，[3]代表female)或者male)
#                 # print(lineArr)
#                 if lineArr[8] == 'white)':
#                     t.write(lineArr[0] + ".jpg " + '0' + "\n")
#                 if lineArr[8] == 'black)':
#                     t.write(lineArr[0] + ".jpg " + '1' + "\n")
#                 if lineArr[8] == 'hispanic)':
#                     t.write(lineArr[0] + ".jpg " + '2' + "\n")
#                 if lineArr[8] == 'asian)':
#                     t.write(lineArr[0] + ".jpg " + '3' + "\n")
#                 if lineArr[8] == 'other)':
#                     t.write(lineArr[0] + ".jpg " + '4' + "\n")
#                 # else:
#                 #     print(lineArr)
#
# t.close()

# with open('./age.txt', 'w') as t:
#     for line in f.readlines():
#         lineArr = line.strip(' ').split(' ')  # strip 去除首尾的指定字符；
#         # lineArr = lineArr.remove('(_sex')
#
#         # num = np.shape(lineArr)[0]
#         if os.path.isfile(os.path.join(images_dir, lineArr[0]) + '.jpg'):
#             if len(lineArr) == 3:
#                 pass
#             else:
#                 lineArr = lineArr  # [0]代表文件名，[1]代表(_sex，[2]空，[3]代表female)或者male)
#                 print(lineArr)
#                 if lineArr[6] == 'senior)':
#                     t.write(lineArr[0] + ".jpg " + '0' + "\n")
#                 if lineArr[6] == 'adult)':
#                     t.write(lineArr[0] + ".jpg " + '1' + "\n")
#                 if lineArr[6] == 'teen)':
#                     t.write(lineArr[0] + ".jpg " + '2' + "\n")
#                 if lineArr[6] == 'child)':
#                     t.write(lineArr[0] + ".jpg " + '3' + "\n")
#
# t.close()




with open('./sex.txt', 'w') as t:
    for line in f.readlines():
        lineArr = line.strip(' ').split(' ')  # strip 去除首尾的指定字符；
        # lineArr = lineArr.remove('(_sex')

        # num = np.shape(lineArr)[0]
        if os.path.isfile(os.path.join(images_dir, lineArr[0]) + '.jpg'):
            if len(lineArr) == 3:
                pass
            else:
                lineArr = lineArr  # [0]代表文件名，[1]代表(_sex，[2]空，[3]代表female)或者male)
                print(lineArr)
                if lineArr[3] == 'female)':
                    t.write(lineArr[0] + ".jpg " + '0' + "\n")
                if lineArr[3] == 'male)':
                    t.write(lineArr[0] + ".jpg " + '1' + "\n")
t.close()


f.close()

