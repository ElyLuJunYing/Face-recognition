import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings

# 禁止打印警告信息
warnings.filterwarnings("ignore")

# 图像大小
m = 128
n = 128

# 计算方差
x_m = np.zeros(3993)
x_e = np.zeros(3993)
img_idx = np.zeros(3993)
ii = 0
for idx in range(1223, 5223):
    img_in_ = cv2.imread(f'dataset/jpgdata/{idx}.jpg', -1)

    if img_in_ is None:
        continue

    # 计算均值及方差
    mean_pixel = np.mean(img_in_.flatten())  # mean_pixel = 0
    error_pixel = np.var(img_in_.flatten(), ddof=1)  # error_pixel = 0

    x_m[ii] = mean_pixel
    x_e[ii] = error_pixel
    img_idx[ii] = idx
    ii += 1

# 箱型图异常值检测 (异常: 过于集中)
plt.figure()
w = 1.5  # 乘数
plt.boxplot(x_e, whis=w)
plt.title("Box graph Outlier detection (abnormal: too concentrated)")
plt.show()

# 计算箱型图异常点 (异常: 过于集中)
# 计算异常点阈值
q1 = np.percentile(x_e, 25)
q3 = np.percentile(x_e, 75)
iqr = q3 - q1
lower_threshold = q1 - w * iqr
upper_threshold = q3 + w * iqr

# 找到异常点 (仅超出下限, 即过于集中的点)
outliers = img_idx[x_e < lower_threshold]  # 超出下限
len_outliers = len(outliers)

# 保存异常点数据为文本文件
outliers_filename = "dataset/outliers_too_dark.txt"
np.savetxt(outliers_filename, outliers, fmt='%d')

# 箱型图异常值检测 (异常: 过亮)
plt.figure()
w = 7  # 乘数
plt.boxplot(x_m, whis=w)
plt.title("Box graph Outlier detection (abnormal: too bright)")
plt.show()

# 计算箱型图异常点 (异常: 过亮)
# 计算异常点阈值
q1 = np.percentile(x_m, 25)
q3 = np.percentile(x_m, 75)
iqr = q3 - q1
lower_threshold = q1 - w * iqr
upper_threshold = q3 + w * iqr

# 找到异常点 (仅超出上限, 即过亮的点)
outliers_too_bright = img_idx[x_m > upper_threshold]  # 超出上限
len_outliers_too_bright = len(outliers_too_bright)

# 保存异常点数据为文本文件
outliers_filename = "dataset/outliers_too_bright.txt"
np.savetxt(outliers_filename, outliers_too_bright, fmt='%d')