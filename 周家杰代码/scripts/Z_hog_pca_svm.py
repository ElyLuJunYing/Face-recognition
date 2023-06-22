# coding=utf-8
from faceIO import FaceRS, readRiIndex, UpdateFaceRS, readFaceRS
from skimage.feature import hog
from sklearn.decomposition import PCA

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
import plot as plot

import time
import numpy as np

def hog_extra_feature():

    # 读取原始图片与索引
    ri, index = readRiIndex()

    img_array_height = ri.shape[0]

    for i in range(ri.shape[0]):
        img = ri[i].reshape((128, 128))

        fd = hog(
            img, orientations=6, block_norm='L1',
            pixels_per_cell=(5, 5), cells_per_block=(2, 2),
            visualize=False,
            transform_sqrt=True,
            multichannel=False
        )

        print('No.' + str(i) + ' shape is ' + str(fd.shape) + '\t\t' + '%.2f' %
              (i * 100 / img_array_height) + '%')

        fd = fd.reshape(1, fd.shape[0])

        if (i == 0):
            face = fd
        elif (i >= 1):
            face = np.concatenate((face, fd), axis=0)

    print('Shape of the result:' + str(face.shape))

    face = np.concatenate([index, face], axis=1)
    
    # 进行HOG特征处理后，划分数据集，并将结果写到目录里
    FaceRS(face, 'hog')

def pca_lower_dimen():

    # 读取已经经过HOG特征处理过的四个文件作为输入
    X_train, X_test, y_train, y_test = readFaceRS('hog')

    X_train_index = X_train[:, 0].reshape(-1, 1)
    X_test_index = X_test[:, 0].reshape(-1, 1)

    X_train = X_train[:, 1:]
    X_test = X_test[:, 1:]

    pca = PCA(n_components=99)
    faceR = pca.fit_transform(X_train)
    faceS = pca.fit_transform(X_test)

    faceR = np.concatenate([X_train_index, faceR], axis=1)
    faceS = np.concatenate([X_test_index, faceS], axis=1)

    print('Shape of faceR: ' + str(faceR.shape))
    print('Shape of faceS: ' + str(faceS.shape))

    # 把HOG+PCA的输出和标签输出到hog_pca目录
    UpdateFaceRS(
        faceR, faceS, y_train, y_test,
        'hog_pca'
    )

def svm_predict():
    # 读取经过上一步特征处理处理的四个输出(hog_pca)
    X_train, X_test, y_train, y_test = readFaceRS('hog_pca')
    # 取age
    y_train = y_train[['age']]
    y_test = y_test[['age']]
    
    print('Start SVM predicting...')

    svm_ovo = OneVsOneClassifier(SVC(kernel='rbf', probability=True))
    svm_ovo.fit(X_train, y_train.values.ravel())

    print('Accuracy score of svm_ovo: ' + '%.3f' %
          svm_ovo.score(X_test, y_test))

    svm_ovr = OneVsRestClassifier(SVC(kernel='rbf', probability=True))
    svm_ovr.fit(X_train, y_train.values.ravel())

    print('Accuracy score of svm_ovr: ' + '%.3f' %
          svm_ovr.score(X_test, y_test))

    plot.conf_matrix(X_test, y_test, svm_ovo, 'hog_pca_svm_ovo')
    plot.conf_matrix(X_test, y_test, svm_ovr, 'hog_pca_svm_ovr')
    plot.plot_roc(X_train, X_test, y_train, y_test,
                  svm_ovr, 'hog_pca_svm_ovr')

if __name__ == '__main__':

    model = 'hog_pca_svm'

    if model == 'hog_pca_svm':
        
        print('Model: hog->pca->svm.')
        startTime = time.time()
        # 用HOG方法进行特征预处理
        hog_extra_feature()
        # 读取HOG特征处理的结果，并用PCA方法进行特征降维
        pca_lower_dimen()
        # 采用SVM进行分类，输出HOG_PCA_SVM犯错矩阵
        svm_predict()
        endTime = time.time()
        print('\nHOG_PCA_SVM costs %.2f seconds.' % (endTime - startTime))

    elif model == 'hog_svm':
        
        print('Model: hog->svm.')
        startTime = time.time()
        # 用HOG方法进行特征预处理
        hog_extra_feature()
        # 采用SVM进行分类，输出HOG_PCA_SVM犯错矩阵
        svm_predict()
        endTime = time.time()
        print('\nHOG_PCA_SVM costs %.2f seconds.' % (endTime - startTime))