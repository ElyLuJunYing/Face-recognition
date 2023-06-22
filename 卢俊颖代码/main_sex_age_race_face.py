import os
import numpy as np
import cv2
import random
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import itertools
import time

###################### function #######################
# 读取图像
def read_img(img_path):
    for i in range(len(img_path)):
        if os.path.isfile(img_path[i]):
            all_images.append(cv2.imread(img_path[i],-1).flatten())

# 卷积核初始化
def weight_init(shape):
    weight = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(weight)

# 偏置项初始化
def bias_init(shape):
    bias = tf.random_normal(shape, dtype=tf.float32)
    return tf.Variable(bias)

# 全连接矩阵初始化
def fch_init(layer1, layer2, const=1):
    min = -const * (6.0 / (layer1 + layer2))
    max = -min
    weight = tf.random_uniform([layer1, layer2], minval=min, maxval=max, dtype=tf.float32)
    return tf.Variable(weight)

# 卷积函数
def conv2d(images, weight):
    return tf.nn.conv2d(images, weight, strides=[1, 1, 1, 1], padding='SAME')

# 最大池化函数
def max_pool2x2(images, tname):
    return tf.nn.max_pool(images, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=tname)

# 显示混淆矩阵
def plot_confuse(truelabel, predictions):
    cm = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    # 指定分类类别
    classes = range(np.max(truelabel)+1)
    title='Normalized Confusion matrix'
   #混淆矩阵颜色风格
    cmap=plt.cm.jet
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
   # 按照行和列填写百分比数据
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Normalized Confusion matrix')
    plt.show()


######################## main #########################
# 选择分类标签
choose = 3  # 性别:0, 年龄:1, 种族:2, 表情:3
num_label = [2, 4, 3, 2]  # 分类数量

# 训练参数设置
train_epochs = 10                     # 最大训练轮数
batch_size = 32                       # 每次训练数据
drop_prob = 0.4                       # 正则化,丢弃比例
learning_rate = [0.001, 0.00000001]   # 学习率

# 数据集路径
data_dir = './dataset/jpgdata'
label_dir = './dataset/label.txt'

# 设置随机种子使结果可复现
random_seed = 42
random.seed(random_seed)  # set random seed for python
np.random.seed(random_seed)  # set random seed for numpy
tf.set_random_seed(random_seed) # tensorFlow 版本 < 2.0
# tf.random.set_seed(random_seed)  # tensorFlow 版本 > 2.0
os.environ['TF_DETERMINISTIC_OPS'] = '1'  # set random seed for tensorflow-gpu

# 读取数据集
data_list = os.listdir(data_dir)
img_path = []
all_images = []
all_labels = []
with open(label_dir, 'r') as f:
    for line in f.readlines():
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()

        path = os.path.join(data_dir,words[0])
        img_path.append(path)

        # 标签独热编码
        if choose == 0:  # 性别
            if words[1] == '0':
                all_labels.append([0, 1])
            elif words[1] == '1':
                all_labels.append([1, 0])

        elif choose == 1:  # 年龄
            if words[2] == '0':
                all_labels.append([0, 0, 0, 1])
            elif words[2] == '1':
                all_labels.append([0, 0, 1, 0])
            elif words[2] == '2':
                all_labels.append([0, 1, 0, 0])
            elif words[2] == '3':
                all_labels.append([1, 0, 0, 0])

        elif choose == 2:  # 种族
            if words[3] == '0':
                all_labels.append([0, 0, 1])
            elif words[3] == '1':
                all_labels.append([0, 1, 0])
            elif words[3] == '2':
                all_labels.append([1, 0, 0])

        elif choose == 3:  # 表情
            if words[4] == '0':
                all_labels.append([0, 1])
            elif words[4] == '1':
                all_labels.append([1, 0])
f.close()

read_img(img_path)

all_images = np.array(all_images)
all_labels = np.array(all_labels)

# 打乱数据集
permutation = np.random.permutation(all_labels.shape[0])
all_images = all_images[permutation,:]
all_labels = all_labels[permutation,:]

# 设置训练集与测试集比例为 8:2
train_total = all_images.shape[0]
train_nums = int(all_images.shape[0] * 0.8)
test_nums = all_images.shape[0] - train_nums
train_images = all_images[0:train_nums, :]  # 训练集
train_labels = all_labels[0:train_nums, :]
test_images = all_images[train_nums:train_total, :]  # 测试集
test_labels = all_labels[train_nums:train_total, :]

# 设置 TensorFlow 输入图像和标签大小
images_input = tf.placeholder(tf.float32,[None,128*128*1],name='input_images')
labels_input = tf.placeholder(tf.float32,[None,num_label[choose]],name='input_labels')
x_input = tf.reshape(images_input,[-1,128,128,1]) # 将图像输入reshape成 128x128x1

# (模型结构) 第 1 层卷积层: 3x3x1x16     
w1 = weight_init([3,3,1,16])
b1 = bias_init([16])
conv_1 = conv2d(x_input,w1)+b1
relu_1 = tf.nn.relu(conv_1,name='relu_1')
max_pool_1 = max_pool2x2(relu_1,'max_pool_1')

# (模型结构) 第 2 层卷积层: 3x3x16x32     
w2 = weight_init([3,3,16,32])
b2 = bias_init([32])
conv_2 = conv2d(max_pool_1,w2) + b2
relu_2 = tf.nn.relu(conv_2,name='relu_2')
max_pool_2 = max_pool2x2(relu_2,'max_pool_2')

# (模型结构) 第 3 层卷积层: 3x3x32x64
w3 = weight_init([3,3,32,64])
b3 = bias_init([64])
conv_3 = conv2d(max_pool_2,w3)+b3
relu_3 = tf.nn.relu(conv_3,name='relu_3')
max_pool_3 = max_pool2x2(relu_3,'max_pool_3')

# (模型结构) 将最终的卷积结果平铺成一维向量
f_input = tf.reshape(max_pool_3, [-1, 16 * 16 * 64])

# (模型结构) 第 1 层全连接层: 16x16x64 X 512
f_w1 = fch_init(16 * 16 * 64, 512)
f_b1 = bias_init([512])
f_r1 = tf.matmul(f_input, f_w1) + f_b1
f_relu_r1 = tf.nn.relu(f_r1) # 激活函数，relu随机丢掉一些权重提供泛化能力
f_dropout_r1 = tf.nn.dropout(f_relu_r1, drop_prob)  # Dropout(正则化)处理, 防止网络出现过拟合的情况

# (模型结构) 第 2 层全连接层:
f_w2 = fch_init(512,128)
f_b2 = bias_init([128])
f_r2 = tf.matmul(f_dropout_r1,f_w2) + f_b2
f_relu_r2 = tf.nn.relu(f_r2)
f_dropout_r2 = tf.nn.dropout(f_relu_r2,drop_prob)

# (模型结构) 输出层
f_w3 = fch_init(128,num_label[choose])
f_b3 = bias_init([num_label[choose]])
f_r3 = tf.matmul(f_dropout_r2,f_w3) + f_b3
f_softmax = tf.nn.softmax(f_r3,name='f_softmax')  # 取结果最大的为分类结果

# 损失函数
cross_entry =  tf.reduce_mean(tf.reduce_sum(-labels_input*tf.log(f_softmax)))  # 交叉熵代价函数

# 优化器, 自动执行梯度下降算法
optimizer = tf.train.AdamOptimizer(learning_rate[0]).minimize(cross_entry)
optimizer2 = tf.train.AdamOptimizer(learning_rate[1]).minimize(cross_entry)

# 计算准确率和损失
arg1 = tf.argmax(labels_input,1)
arg2 = tf.argmax(f_softmax,1)
cos = tf.equal(arg1,arg2)
acc = tf.reduce_mean(tf.cast(cos,dtype=tf.float32))  # tf.cast将bool值转换为浮点数

# 开始训练模型 (CPU)
init = tf.global_variables_initializer()
sess = tf.Session() # tf.Session( config=tf.ConfigProto(log_device_placement=True) )
sess.run(init)
Cost = []
Accuracy = []
print("Training...")
for i in range(train_epochs):  # 遍历每个epoch
    acc_array = []
    cross_array = []

    if i < 8:  # 8
        optimizer_choose = optimizer
    elif Accuracy[-1] - Accuracy[-2] > 0.005:
        optimizer_choose = optimizer2  # 降低学习率
    else:
        break  # 模型收敛

    # 遍历每一个batch
    for idx in range(0, int(len(train_images)/batch_size)):
        train_input = train_images[idx*batch_size:(idx+1)*batch_size]
        train_labels_input = train_labels[idx*batch_size:(idx+1)*batch_size]

        result, acc1, cross_entry_r, cos1, f_softmax1, relu_1_r = sess.run(
            [optimizer_choose, acc, cross_entry, cos, f_softmax, relu_1],
            feed_dict={images_input: train_input, labels_input: train_labels_input})
        acc_array.append(acc1)
        cross_array.append(cross_entry_r)

    print("epoch:", i, " train accuracy:", np.mean(acc_array))
    Cost.append(np.mean(cross_array))
    Accuracy.append(np.mean(acc_array))

# 绘制模型训练时的损失函数曲线
fig1, ax1 = plt.subplots(figsize=(10, 7))
plt.plot(Cost)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Cost')
plt.title('Cross Loss')
plt.grid()
plt.savefig('Train Cross Loss.png')
plt.show()

# 绘制模型训练时的准确率曲线
fig7, ax7 = plt.subplots(figsize=(10, 7))
plt.plot(Accuracy)
ax7.set_xlabel('Epochs')
ax7.set_ylabel('Accuracy Rate')
plt.title('Train Accuracy Rate')
plt.grid()
# plt.ylim(0,1)
plt.savefig('Train Accuracy Rate.png')
plt.show()

# 使用测试集进行验证并计时
tic = time.time()
arg2_r = sess.run(arg2,feed_dict={images_input:test_images,labels_input:test_labels})  # y_pred
toc = time.time()
test_time = toc-tic
print("每张图像的处理耗时:", test_time/test_nums, "s")  # 打印模型处理每张图像的耗时
arg1_r = sess.run(arg1,feed_dict={images_input:test_images,labels_input:test_labels})  # y_true
print("测试集分类报告:")
print(classification_report(arg1_r, arg2_r))  # 打印分类报告
print("混淆矩阵:")
print(confusion_matrix(arg1_r, arg2_r)/test_nums)  # 打印混淆矩阵
plot_confuse(arg1_r, arg2_r)

# #保存模型
# saver = tf.train.Saver()
# saver.save(sess, './model/face_recognition')