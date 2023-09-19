# -*- coding: utf-8 -*-
"""
@File: eg_unsw.py
@Time: 2023/09/18 21:24:25
@Author: Chaoqun Guo <chaoqunguo317@gmail.com>
@Version: 0.0.1
@Desc: 
"""

from scipy.optimize import curve_fit
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


def make_imb_data(num_max, num_classes, imb_ratio):
    mu = np.power(1 / imb_ratio, 1 / (num_classes - 1))
    class_num_list = []
    for i in range(num_classes):
        if i == (num_classes - 1):
            class_num_list.append(int(num_max / imb_ratio))
        else:
            class_num_list.append(int(num_max * np.power(mu, i)))
    num_per_class = class_num_list
    return list(num_per_class)


num_per_class = [37000, 18871, 11132, 6062, 4089, 3496, 677, 583, 378, 44]
index = ["Normal", "Generic", "Exploits", "Fuzzers", "DoS",
         "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"]
colors = ["g", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r"]

pie_data = [num_per_class[0], sum(num_per_class[1:])]
print(pie_data)
pie_labels = ['Normal', 'Abnormal']
pie_colors = colors
total = sum(pie_data)
percentage = [(val/total)*100 for val in pie_data]

plt.figure(figsize=(15, 12))
# fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(15,12))
plt.subplot(131)
plt.bar(index, num_per_class, align='center', width=0.5, color=colors)
plt.xlabel(
    r"(a) UNSW_NB15 dataset distribution, where $n_{normal} \gg n_{worms}$.")
plt.ylabel("Training Samples.")
plt.yscale("linear")
plt.xticks(rotation=45)

plt.subplot(132)
plt.pie(percentage, colors=pie_colors,
        autopct='')
plt.legend(pie_labels, loc="upper center",)

plt.subplot(133)
# plt.pie(percentage, colors=pie_colors,
# autopct='')
# plt.legend(pie_labels, loc="right", bbox_to_anchor=(0, 0.85))
# plt.bar(index, num_per_class, align='center', width=0.8, color=colors)
# plt.xlabel(
#     r"UNSW_NB15 dataset distribution, where $n_{normal} \gg n_{worms}$.")
# plt.ylabel("Training Samples.")
# plt.xticks(rotation=15)
plt.show()
