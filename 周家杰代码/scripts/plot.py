from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
import file_path as dirt
from scipy import interp
import numpy as np

def conf_matrix(X_test, y_test, classifier, method_name):
    # 定义类别名称
    class_names = ['child', 'teen', 'adult', 'senior']

    # 绘制归一化的混淆矩阵
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize='all')
    disp.ax_.set_title('Normalized confusion matrix')

    # 获取混淆矩阵
    confusionMatrix = disp.confusion_matrix
    print('Confusion Matrix:\n', confusionMatrix)
    
    # 保存混淆矩阵的图像
    plt.savefig(dirt.Dirt.pic_path+method_name+'_confM.jpg')

    # 计算错误矩阵并可视化
    row_sums = np.sum(confusionMatrix, axis=1)  # 求行和
    err_matrix = confusionMatrix / row_sums  # 求错误所占百分比
    np.fill_diagonal(err_matrix, 0)
    print('Error Matrix:\n', err_matrix)

    # 绘制错误矩阵的灰度图并保存图像
    plt.matshow(err_matrix, cmap=plt.cm.gray)
    plt.savefig(dirt.Dirt.pic_path+method_name + '_errM.jpg')

    # 使用分类器对测试集进行预测，并生成分类报告（precision、recall、f1_score等指标）
    y_pred = classifier.predict(X_test)
    print('Report:\n', classification_report(y_test, y_pred))

def plot_roc(X_train, X_test, y_train, y_test, classifier, method_name):
    n_classes = 4

    # onehot 编码
    class_names = ['child', 'teen', 'adult', 'senior']
    y_train = label_binarize(y_train, classes=class_names)  # 将训练集标签进行one-hot编码
    y_test = label_binarize(y_test, classes=class_names)    # 将测试集标签进行one-hot编码

    X = np.concatenate([X_train, X_test])  # 将训练集特征和测试集特征合并
    y = np.concatenate([y_train, y_test])  # 将训练集标签和测试集标签合并
    y_score = classifier.predict_proba(X_test)  # 获取分类器在测试集上的预测概率
    # y_score = classifier.fit(X_train, y_train).decision_function(X_test)  # 另一种获取分类器决策函数得分的方法
    print(y_score.shape)

    # 计算每个类别的ROC曲线和ROC面积
    fpr = dict()  # 伪阳性率
    tpr = dict()  # 真阳性率
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])  # 计算每个类别的伪阳性率、真阳性率和阈值
        roc_auc[i] = auc(fpr[i], tpr[i])  # 计算每个类别的ROC面积

    # 计算微平均的ROC曲线和ROC面积
    fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_score.ravel())  # 计算微平均的伪阳性率、真阳性率和阈值
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])  # 计算微平均的ROC面积

    # 计算宏平均的ROC曲线和ROC面积
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))  # 将所有类别的伪阳性率拼接并去重
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])  # 利用一维插值计算宏平均的真阳性率
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])  # 计算宏平均的ROC面积

    # 绘制所有ROC曲线
    plt.figure()
    lw = 2
    plt.plot(fpr['micro'], tpr['micro'],
             label='micro-average ROC curve (area = {0:0.2f})'
             ''.format(roc_auc['micro']),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr['macro'], tpr['macro'],
             label='macro-average ROC curve (area = {0:0.2f})'
             ''.format(roc_auc['macro']),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc='lower right')
    plt.savefig(dirt.Dirt.pic_path + method_name + '_roc.jpg')