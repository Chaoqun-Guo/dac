# -*- coding: utf-8 -*-
"""
@File: help.py
@Time: 2023/08/22 20:58:50
@Author: Chaoqun Guo <chaoqunguo317@gmail.com>
@Version: 0.0.1
@Desc: help to understand
"""
# import numpy as np
# import matplotlib.pyplot as plt

# # 设置均值和标准差
# mean = 0
# stddev = 1

# # 生成一组 x 值
# x = np.linspace(-5, 5, 1000)

# # 计算高斯分布的概率密度函数
# pdf = (1 / (stddev * np.sqrt(2 * np.pi))) * \
#     np.exp(-0.5 * ((x - mean) / stddev)**2)

# # 绘制高斯分布曲线
# plt.figure(figsize=(8, 6))
# plt.plot(x, pdf, color='blue', label='Gaussian Distribution')
# plt.xlabel('x')
# plt.ylabel('Probability Density')
# plt.title('Gaussian Distribution')
# plt.legend()
# plt.grid(True)
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import multivariate_normal

# # 均值和协方差矩阵
# mean = np.array([2, 3])
# covariance = np.array([[1, 0.5], [0.5, 2]])

# # 创建一个网格
# x, y = np.meshgrid(np.linspace(-5, 10, 100), np.linspace(-5, 10, 100))
# xy = np.column_stack([x.flat, y.flat])

# # 计算多元高斯分布的概率密度值
# pdf_values = multivariate_normal.pdf(xy, mean=mean, cov=covariance)
# pdf_values = pdf_values.reshape(x.shape)

# # 绘制高斯分布概率密度函数曲线
# plt.figure(figsize=(8, 6))
# plt.contourf(x, y, pdf_values, cmap='viridis')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Gaussian Distribution')
# plt.colorbar()
# plt.show()

# import numpy as np

# # 示例数据，每列代表一个随机变量，每行代表一个样本
# data = np.array([[1, 1,],
#                  [2, 4,],
#                  [3, 5,],
#                  [4, 6,]])
# print(data.shape)
# x_ = data[:, 0].mean()
# y_ = data[:, 1].mean()
# print(x_, y_)
# # 计算协方差矩阵
# cov_matrix = np.cov(data, rowvar=False)

# print("Covariance Matrix:")
# print(cov_matrix)


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.mixture import GaussianMixture

# # 生成两个高斯分布的样本
# np.random.seed(0)
# n_samples = 100

# # 生成两个高斯分布的数据
# X1 = np.random.randn(n_samples, 2) + np.array([2, 2])
# X2 = np.random.randn(n_samples, 2) + np.array([-2, -2])
# X = np.vstack([X1, X2])

# # 使用 GMM 拟合数据
# n_components = 4  # 设定高斯分布的个数
# gmm = GaussianMixture(n_components=n_components)
# gmm_out = gmm.fit(X)
# print(gmm_out.predict_proba(X))
# print(gmm_out.predict(X))


# # 生成新的样本
# n_samples_gen = 100
# X_gen, _ = gmm.sample(n_samples_gen)

# # 绘制生成的样本
# plt.scatter(X[:, 0], X[:, 1], c='blue', marker='o', label='Original Data')
# plt.scatter(X_gen[:, 0], X_gen[:, 1], c='red',
#             marker='x', label='Generated Data')
# plt.legend()
# plt.title('Gaussian Mixture Model Example')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from collections import Counter
# 生成二维的两类数据
n_samples = 100
mean1 = [2, 2]
cov1 = [[1, 0.5], [0.5, 1]]
data1 = np.random.multivariate_normal(mean1, cov1, n_samples)

mean2 = [-2, -2]
cov2 = [[1, -0.5], [-0.5, 1]]
data2 = np.random.multivariate_normal(mean2, cov2, n_samples)

mean3 = [-3, -5]
cov3 = [[1, -0.5], [-0.5, 1]]
data3 = np.random.multivariate_normal(mean3, cov3, n_samples)

# 合并数据
X = np.concatenate((data1, data2, data3), axis=0)
Y = np.concatenate(([0]*100, [1]*100, [2]*100), axis=0)

# 定义高斯混合模型并拟合数据
n_components = 2
gmm = GaussianMixture(n_components=n_components,
                      random_state=0, max_iter=10000)
gmm.fit(X)

y_hat = gmm.predict(X)
y_com = gmm.predict_proba(X)

print(y_hat)
n_comp = [[] for _ in range(n_components)]
print(n_comp)
for y_, p_y in zip(Y, y_hat):
    n_comp[p_y].append(y_)

print(n_comp)
purity = [dict(Counter(i)) for i in n_comp]
for idx, n_purity in enumerate(purity):
    total = sum(n_purity.values())
    for k, v in n_purity.items():
        n_purity[k] = [round(v/total, 4), total]
    purity[idx] = n_purity
print(purity)
n_comp_purity = [np.max([i[0] for i in list(pu.values())]) for pu in purity]
print(n_comp_purity)
filter_comp = [i > 0.9 for i in n_comp_purity]
print(filter_comp)

# 绘制数据点
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.scatter(X[:100, 0], X[:100, 1], s=30, c='green',
            marker='o', alpha=0.5, label='C-1')
plt.scatter(X[-200:-100, 0], X[-200:-100, 1], s=30, c='red',
            marker='x', alpha=0.5, label='C-2')
plt.scatter(X[-5:, 0], X[-5:, 1], s=30, c='blue',
            marker='^', alpha=0.5, label='C-3')
# plt.title('Gaussian Mixture Model')
# plt.xlabel('Feature 1')
plt.xlabel('(a)')
plt.legend()


def draw_ellipse(position, covariance, ax=None, c=None, **kwargs):
    """绘制高斯分布的椭圆"""
    ax = ax or plt.gca()

    # 计算椭圆的主轴和角度
    v, w = np.linalg.eigh(covariance)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180.0 * angle / np.pi

    # 创建椭圆并添加到图中
    ell = Ellipse(position, v[0], v[1], angle=(180.0 +
                  angle), alpha=0.5, color=c, **kwargs)
    ax.add_patch(ell)
    return ell


# 绘制高斯分量
for i, c in zip(range(n_components), ['red', 'green', 'blue', 'yellow']):
    mean = gmm.means_[i]
    cov = gmm.covariances_[i]
    draw_ellipse(mean, cov, c=c)

plt.subplot(1, 2, 2)
plt.scatter(y_com[:100, 0], y_com[:100, 1], s=30, c='green',
            marker='o', alpha=0.5, label='C-G-1')
plt.scatter(y_com[-200:-100, 0], y_com[-200:-100, 1], s=30, c='red',
            marker='x', alpha=0.5, label='C-G-2')
plt.scatter(y_com[-100:, 0], y_com[-100:, 1], s=30, c='blue',
            marker='^', alpha=0.5, label='C-G-3')

# plt.title('Gaussian Mixture Model with n_components')
# plt.xlabel('Feature 2')
plt.xlabel('(b)')
plt.tight_layout()
plt.legend()
plt.show()
