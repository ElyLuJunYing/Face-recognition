import os
import io
import numpy as np

Name_label = [] #姓名标签
path = './face/'   #数据集文件路径
dir = os.listdir(path)  #列出所有人
label = 0   #设置计数器

#数据写入
with open('./train.txt','w') as f:
    for name in dir:
        Name_label.append(name)
        print(Name_label[label])
        after_generate = os.listdir(path +'\\'+ name)
        for image in after_generate:
            if image.endswith(".png"):
                f.write(image + ";" + str(label)+ "\n")
        label += 1
#########################################################################
 # 打开数据集的txt
    with open(r".\train.txt","r") as f:
        lines = f.readlines()
    #打乱数据集
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    # 80%用于训练，20%用于测试。
    num_val = int(len(lines)*0.2)   #
    num_train = len(lines) - num_val  #4715
