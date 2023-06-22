#说明：由于程序需要调用人脸特征识别器以及保存和读取数据文件，代码py文件应放入Face_recognition文件夹下运行。Face_recognition下的test_data文件夹存放测试用的照片，training_data用于存放训练人脸模型用的照片数据。程序运行过程中请按控制台提示操作。
#提示：若非必要请勿使用降噪功能降噪，降噪会降低照片的画质，影响人脸识别。

import os
import cv2
import numpy as np
from alive_progress import alive_bar
from tkinter import filedialog
import time

 #定义函数：调用摄像头进行人脸识别
def video_id():
    cap = cv2.VideoCapture(700)
    time.sleep(0.1)

    # 调用OpenCV人脸识别分类器
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # 识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)

    ret, img = cap.read()

    while cap.isOpened():
        k = 1
        while k == 1: #开启摄像头后每隔一段时间拍摄一张照片进行实时识别
            cv2.namedWindow("1", cv2.WINDOW_NORMAL)
            cv2.imshow("1", img)
            cv2.imwrite('camera.jpg', img)
            k = k + 1
        label = img_id2() #获得摄像头内人脸对应的人名

        ret, img = cap.read()
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #灰度转换

        # scaleFactor=1.2,图片缩放比例,默认是1.1
        # minNeighbors=3：匹配成功所需要的周围矩形框的数目，每一个特征匹配到的区域都是一个矩形框，只有多个矩形框同时存在的时候，才认为是匹配成功，比如人脸，这个默认值是3
        # minSize:匹配物体的大小范围
        faceRects = faceCascade.detectMultiScale(grey, scaleFactor=2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:  # 大于0则检测到人脸
            for faceRect in faceRects:  # 单独框出每一张人脸
                x, y, w, h = faceRect
                cv2.rectangle(img, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2) #绘制矩形框出人脸
                cv2.putText(img, label, (x,y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0),2) #标识人脸对应的人名
        cv2.namedWindow("1", cv2.WINDOW_NORMAL)
        cv2.imshow("1", img)
        # 等待1ms显示图像，若过程中按“q”退出
        if cv2.waitKey(1) == ord('q'):
            break
    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()

#定义函数：检测人脸
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#灰度转化
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#加载Haar特征分类器
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)# 检测多尺度图像，返回脸部区域信息列表（x，y，宽，高）
    if (len(faces)) == 0:
        return None, None #若未检测到脸部，则返回原始图像
    (x, y, w, h) = faces[0] #x，y维左上角坐标，w，h为矩形宽高
    return gray[y:y + w, x:x + h], faces[0] #返回灰度图像脸部部分以及坐标信息

#定义函数：判断是否能检测出人脸，用于检测一照片内多人脸的情况
def detect_face2(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#灰度转化
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#加载Haar特征分类器
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)# 检测多尺度图像，返回脸部区域信息列表（x，y，宽，高）
    if (len(faces)) == 0:#检查是否检测到脸部，返回k值用于接下来的判断
        k = 0
    else:
        k = 1
    return k

#定义函数：该函数中调用了detect_face函数，读取所有训练图像，从每个图像中检测出人脸
#并返回两个相同大小的列表，分分别为脸部信息和标签
def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)#获取数据文件夹中的目录
    faces = [] #创建两个空列表用于存放脸部数据和标签
    labels = []
    for dir_name in dirs:
        #浏览每个目录并访问其中的图像
        label = int(dir_name) #文件名对应标签
        subject_dir_path = data_folder_path + '/' + dir_name #建立包含当前主题图像的目录路径
        subject_images_names = os.listdir(subject_dir_path) #获取给定主题目录内的图像名称
        print(f"读取{subjects[label]}图片中")
        with alive_bar(len(subject_images_names), force_tty=True) as bar:
            for image_name in subject_images_names: #浏览每张图片并检测脸部，然后将脸部信息添加到列表faces
                image_path = subject_dir_path + '/' + image_name #建立图像路径
                image = cv2.imread(image_path) #读取图像
                face, rect = detect_face(image)#调用detect_face函数检测脸部
                if face is not None: #只有当检测到脸部时才将脸部信息添加到列表并添加相应的标签
                    faces.append(face)
                    labels.append(label)
                bar()
    return faces, labels #返回脸部信息的列表和标签的列表，脸部信息和标签一一对应

#定义函数：绘制矩形——根据给定的（x,y）坐标和宽度在图像上绘制矩形
def draw_rectangle(img,rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h),(0, 255, 0), 3)

#定义函数：绘制填充矩形——用于绘制填充矩形屏蔽已检测出的人脸
def draw_rectangle2(img,rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h),(0, 255, 0), -1)

#定义函数：标识人名——根据给定的（x，y）坐标标识出人脸对应的人名
def draw_text(img,text,x,y,z):
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_COMPLEX, z, (0, 255, 0),2)

#定义函数：识别照片中的人物，找到与输入的人脸照片接近的模型并进行比对，识别后在脸部周围绘制一个矩形及标识出人名
def predict(test_img):
    img = test_img.copy() #生成图像的副本，这样就能保留原始图像
    img2 = test_img.copy()
    k2 = 1
    while k2 == 1:
        face, rect = detect_face(img2) #检测人脸
        k2 = detect_face2(img2)
        if k2 == 0:
            break
        else:
            label, confidence= face_recognizer.predict(face) #找处与所输入图像最接近的人脸模型，并返回人脸模型对应的标签以及置信值confidence
            if confidence > 70: #【更改处】程序找出一个与所输入图像最接近的模型，并得出置信值confidence，值越小表示模型越接近于人脸
                label_text = "unknown" #当confidence的值大于70则表示模型与所输入人脸照片差异过大，此时显示无法识别人脸“unknown”
                draw_text(img, label_text, rect[0], rect[1] - 5, 3) #绘制文本
            else:
                label_text = subjects[label] #当confidence小于70时，表示人脸模型与所输入人脸照片差异在一定范围内，此时根据标签标识出对应的人名
                draw_text(img, label_text, rect[0], rect[1] - 5, 3) #绘制文本
            print(f'置信值为:{confidence}')
            print('识别完成')
            draw_rectangle(img, rect) #绘制矩形
            draw_rectangle2(img2, rect)#屏蔽已检测出的人脸，用于检测其他人脸
    return img

#定义函数：用于摄像头识别人脸，返回人脸对应的人名
def predict2(test_img):
    img = test_img.copy() #生成图像的副本，这样就能保留原始图像
    img2 = test_img.copy()
    global label_text
    face, rect = detect_face(img2) #检测人脸
    k2 = detect_face2(img2)
    if k2 != 0:
        label, confidence= face_recognizer.predict(face) #找处与所输入图像最接近的人脸模型，并返回人脸模型对应的标签以及置信值confidence
        if confidence > 50: #【更改处】程序找出一个与所输入图像最接近的模型，并得出置信值confidence，值越小表示模型越接近于人脸
            label_text = "unknown" #当confidence的值大于70则表示模型与所输入人脸照片差异过大，此时显示无法识别人脸“unknown”
        else:
            label_text = subjects[label] #当confidence小于70时，表示人脸模型与所输入人脸照片差异在一定范围内，此时根据标签标识出对应的人名
    else:
        label_text = ''
    return label_text

#定义函数：用于输入照片进行识别
def img_id():
    Filepath = filedialog.askopenfilename()
    print('请选择一张照片：')
    global predicted_img2
    test_img2 = cv2.imread(Filepath) #输入一张图片
    choose_3 = input('是否进行降噪处理？（1.是 2.否）：')
    if choose_3 == '1':
        result = cv2.blur(test_img2, (5, 5))
        print('降噪处理完成')
        predicted_img2 = predict(result)
    elif choose_3 == '2' :
        predicted_img2 = predict(test_img2) #调用函数predict识别人脸

    cv2.namedWindow('result', cv2.WINDOW_NORMAL) #创建一个窗口用于显示图片

    cv2.imshow('result', predicted_img2) #窗口名称
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#定义函数：用于摄像头启动后拍摄照片识别人脸
def img_id2():
    test_img2 = cv2.imread('camera.jpg') #输入一张图片
    label= predict2(test_img2) #调用函数predict识别人脸
    return label #返回人脸对应的人名

#定义函数：用于对照片数据进行去噪处理
def blur(data_folder_path):
    dirs = os.listdir(data_folder_path)  # 获取数据文件夹中的目录
    for dir_name in dirs:
        # 浏览每个目录并访问其中的图像
        subject_dir_path = data_folder_path + '/' + dir_name  # 建立包含当前主题图像的目录路径
        subject_images_names = os.listdir(subject_dir_path)  # 获取给定主题目录内的图像名称
        print(f"正在对照片进行去噪处理")
        with alive_bar(len(subject_images_names), force_tty=True) as bar:
            for image_name in subject_images_names:  # 浏览每张图片并检测脸部，然后将脸部信息添加到列表faces
                image_path = subject_dir_path + '/' + image_name  # 建立图像路径
                image = cv2.imread(image_path)  # 读取图像
                result = cv2.blur(image, (5, 5)) # 均值滤波去噪
                address_1 = subject_dir_path + '/' + image_name #获得将去噪后的照片的地址及文件名
                cv2.imwrite(address_1, result) #保存去噪后的数据文件
                bar() #进度显示
    print('去噪处理完成')

subjects = ['Panjinlong', 'Xiewei', 'Chenziyuan', 'Chenkehan', 'Yijiaming', 'Lijiahang'] #每组照片训练出的模型所对应的人名

bool = input('输入新数据或读取已有数据？（1.输入新数据 2.读取已有数据）：')
if bool == '1':
    bool3 = input('是否对数据照片进行去噪处理？（1.是 2.否）：')
    if bool3 == '1':
        print('请选择照片所在路径：')
        data_folder_path = filedialog.askdirectory()
        blur(data_folder_path) #调用函数进行降噪处理
    print('请选择人脸数据用于人脸模型的训练（选择training_data文件夹）：')
    folderpath = filedialog.askdirectory()
    faces, labels = prepare_training_data(folderpath) #从指定路径输入照片进行人脸检测，返回人脸的位置信息和对应的标签
    bool2 = input('数据已输入，是否保存？（1.是 2.否）:')
    if bool2 == '1':
        np.save('faces.npy',faces) #保存人脸位置信息于指定路径的文件中
        np.save('labels.npy',labels) #保存人脸对应标签于指定的路径文件中
        print('数据文件已保存在Face_recognization文件夹下')
elif bool == '2':
    faces = np.load('faces.npy',allow_pickle=True) #读取指定路径文件中的人脸位置信息
    labels = np.load('labels.npy') #读取指定路径文件中的人脸标签信息
    print('数据读取成功')
print('正在训练人脸模型')
face_recognizer = cv2.face.LBPHFaceRecognizer_create() #创建LBPH识别器，用于人脸识别
face_recognizer.train(faces, np.array(labels)) #利用人脸位置信息以及相应的标签，调用创建的LBPH识别器进行人脸模型训练
print('模型训练完成')

while True:
    choose = input('请选择识别方式：1.启动摄像头识别 2.输入照片识别 3.退出:')
    if choose == '1':
        print("摄像头已启动，请按'q'键退出")
        video_id()
    elif choose == '2':
        img_id()
    elif choose == '3':
        break