import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
# 定义模型名称和对应的文件路径
model_names = ['CNN','Logistic Regression','LSTM','Proposed Model','DNN', 'RF', 'Support Vector Machine']
file_paths = [f'{name}_results.csv' for name in model_names]

# 初始化图形
plt.figure(figsize=(10, 8))

# 遍历每个模型
for file_path, model_name in zip(file_paths, model_names):
    # 读取保存的标签文件
    results_df = pd.read_csv(file_path)

    # 清理标签字符串并转换为实际数组
    true_labels = results_df['True Labels'].str.replace(' ', ',').apply(eval).tolist()
    predicted_labels = results_df['Predicted Labels'].str.replace(' ', ',').apply(eval).tolist()

    # 转换为 NumPy 数组
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # 初始化 FPR 和 TPR 的列表
    all_fpr = np.linspace(0, 1, 100)
    mean_tpr = 0.0
    num_classes = true_labels.shape[1]

    # 计算每个类的 ROC 曲线并累计 TPR
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(true_labels[:, i], predicted_labels[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)

    mean_tpr /= num_classes  # 计算均值 TPR
    mean_auc = auc(all_fpr, mean_tpr)  # 计算均值 AUC

    # 绘制均值 ROC 曲线
    plt.plot(all_fpr, mean_tpr, label=f'{model_name} (AUC = {mean_auc:.2f})')

# 绘制对角线
plt.plot([0, 1], [0, 1], 'k--', lw=2)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend(loc='lower right')
plt.show()