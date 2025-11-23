import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, \
    confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号 '-' 显示为方块的问题

# 加载数据集
data = pd.read_excel('../500例IgA数据集.xlsx', sheet_name='data')

# 检查并处理NaN和无穷大值
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# 定义输入特征和目标变量
features = data[['年龄', '性别', '尿蛋白（≥1g）', '尿蛋白定量', '血尿（含镜下）', '镜检红细胞', '头晕，血压高', '收缩压', '舒张压',
                 '肾功能重度下降(GFR＜30）', '肾小球滤过率（MDRD）', '口气重，呼气时有尿臭', '疲惫/困乏/乏力', '易感冒',
                 '自汗/盗汗', '恶风', '皮肤瘙痒', '肌肉/头身/肢节酸楚', '水肿加重', '腰酸', '气短懒言',
                 '夜尿增多（夜间睡眠时尿量＞750ml或大于白天尿量）', '手足心热', '目睛干涩', '咽干咽燥',
                 '腰痛固定', '久病(病程≥3个月）', '皮肤瘀斑、瘀点/肌肤甲错/皮肤赤丝红缕/蟹爪纹络', '肢体麻木',
                 '面色黧黑', '急躁易怒', '头痛 ', '视物模糊甚则黑蒙', '震颤，搐搦', '纳呆、泛恶',
                 '面色不华', '畏寒怕冷']]
targets = data[['肾虚（肾气阴）证', '风湿证', '瘀痹证', '肝风证', '溺毒证']]

# 对数值型和分类型数据进行预处理
numerical_cols = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = features.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),  # 数值型特征进行标准化
        ('cat', OneHotEncoder(), categorical_cols)  # 分类型特征进行One-Hot编码
    ])

X_processed = preprocessor.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(X_processed, targets, test_size=0.2, random_state=42)

# 定义随机森林模型
model = RandomForestClassifier(n_estimators=1, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 评估模型
y_pred_proba = model.predict_proba(X_test)  # 预测概率
y_pred = model.predict(X_test)  # 预测类别

total_accuracy = 0
total_precision = 0
total_recall = 0
total_f1 = 0
total_auc = 0
num_labels = len(targets.columns)

# 对每个标签分别计算评估指标
for i, label in enumerate(targets.columns):
    accuracy = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
    precision = precision_score(y_test.iloc[:, i], y_pred[:, i])
    recall = recall_score(y_test.iloc[:, i], y_pred[:, i])
    f1 = f1_score(y_test.iloc[:, i], y_pred[:, i])
    auc = roc_auc_score(y_test.iloc[:, i], y_pred_proba[i][:, 1])

    print(f'{label} - 准确率: {accuracy:.3f}, 精确率: {precision:.3f}, 召回率: {recall:.3f}, '
          f'F1得分: {f1:.3f}, AUC: {auc:.3f}')

    total_accuracy += accuracy
    total_precision += precision
    total_recall += recall
    total_f1 += f1
    total_auc += auc

    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_test.iloc[:, i], y_pred_proba[i][:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='AUC = %0.2f' % auc)
    plt.plot([0, 1], [0, 1], '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率',fontsize=24)
    plt.ylabel('真正率',fontsize=24)
    plt.title(f'{label} - ROC曲线',fontsize=26, fontweight='bold')
    plt.legend(loc="lower right",fontsize=22)
    plt.savefig(f'rf{label}.png')
    plt.close()

    # 绘制混淆矩阵
    cm = confusion_matrix(y_test.iloc[:, i], y_pred[:, i])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('预测标签',fontsize=24)
    plt.ylabel('真实标签',fontsize=24)
    plt.title(f'{label} - 混淆矩阵',fontsize=26, fontweight='bold')
    plt.savefig(f'rf_matrix_{label}.png')
    plt.close()

# average_accuracy = total_accuracy / num_labels
# average_precision = total_precision / num_labels
# average_recall = total_recall / num_labels
# average_f1 = total_f1 / num_labels
# average_auc = total_auc / num_labels
#
# print(f'\n平均指标 - 准确率: {average_accuracy:.3f}, 精确率: {average_precision:.3f}, 召回率: {average_recall:.3f}, '
#       f'F1得分: {average_f1:.3f}, AUC: {average_auc:.3f}')


# # 初始化ROC曲线的图形
# plt.figure(figsize=(10, 8))

# # 对每个标签分别计算评估指标并绘制ROC曲线
# for i, label in enumerate(targets.columns):
#     accuracy = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
#     precision = precision_score(y_test.iloc[:, i], y_pred[:, i])
#     recall = recall_score(y_test.iloc[:, i], y_pred[:, i])
#     f1 = f1_score(y_test.iloc[:, i], y_pred[:, i])
#     auc = roc_auc_score(y_test.iloc[:, i], y_pred_proba[i][:, 1])
#
#     print(f'{label} - 准确率: {accuracy:.3f}, 精确率: {precision:.3f}, 召回率: {recall:.3f}, '
#           f'F1得分: {f1:.3f}, AUC: {auc:.3f}')
#
#     total_accuracy += accuracy
#     total_precision += precision
#     total_recall += recall
#     total_f1 += f1
#     total_auc += auc
#
#     # 绘制每个类别的ROC曲线在同一个图上
#     fpr, tpr, _ = roc_curve(y_test.iloc[:, i], y_pred_proba[i][:, 1])
#     plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.2f})')
#
# # 绘制对角线
# plt.plot([0, 1], [0, 1], 'r--')
#
# # 设置图形标题和标签
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('假阳性率')
# plt.ylabel('真正率')
# plt.title('所有类别的ROC曲线')
# plt.legend(loc="lower right")
# plt.grid(True)
#
# # 保存汇总的ROC曲线图
# plt.savefig('rf_combined_roc_curve.png')
# plt.show()
#
# # 打印平均评估指标
# average_accuracy = total_accuracy / num_labels
# average_precision = total_precision / num_labels
# average_recall = total_recall / num_labels
# average_f1 = total_f1 / num_labels
# average_auc = total_auc / num_labels
#
# print(f'\n平均指标 - 准确率: {average_accuracy:.3f}, 精确率: {average_precision:.3f}, 召回率: {average_recall:.3f}, '
#       f'F1得分: {average_f1:.3f}, AUC: {average_auc:.3f}')

# # 初始化用于计算平均ROC曲线的数据
# all_fpr = np.linspace(0, 1, 100)
# mean_tpr = 0.0
# # # 用于保存每个标签的FPR和TPR
# # fpr_list = []
# # tpr_list = []
# # 对每个标签分别计算ROC曲线并累计TPR
# for i, label in enumerate(targets.columns):
#     fpr, tpr, _ = roc_curve(y_test.iloc[:, i], y_pred_proba[i][:, 1])
#     # fpr_list.append(fpr)  # 保存FPR
#     # tpr_list.append(tpr)  # 保存TPR
#     mean_tpr += np.interp(all_fpr, fpr, tpr)  # 插值计算TPR
#
#     mean_tpr[0] = 0.0  # 确保第一个点是(0,0)
#
# # 计算平均TPR
# mean_tpr /= num_labels
# mean_tpr[-1] = 1.0  # 确保最后一个点是(1,1)
#
# # 计算宏平均AUC
# mean_auc = auc(all_fpr, mean_tpr)
# # 保存到文件或进一步处理
# # np.save('rf_mean_tpr.npy', mean_tpr)
# # np.save('fr_mean_fpr.npy', mean_fpr)
#
# # 确保将 fpr 和 tpr 单独存储为列，而不是将它们打包成列表
# roc_data = pd.DataFrame({
#     'fpr': all_fpr,  # 直接使用 all_fpr 数组
#     'tpr': mean_tpr,  # 使用计算后的 mean_tpr 数组
# })
#
# # 单独保存 AUC 值（如需要保存为 CSV 文件中的附加信息）
# # auc_value = pd.DataFrame({'auc': [mean_auc]})
# # 假设 mean_auc 是您的 AUC 计算结果
# roc_data['auc'] = [mean_auc] * len(roc_data)
#
# # 保存到文件
# roc_data.to_csv('rf_roc_data.csv', index=False)
# # auc_value.to_csv('rf_auc_value.csv', index=False)
# # 绘制宏平均ROC曲线
# plt.figure(figsize=(10, 8))
# plt.plot(all_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.3f})', lw=2)
#
# # 绘制对角线
# plt.plot([0, 1], [0, 1], 'r--')
#
# # 设置图形标题和标签
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('假阳性率')
# plt.ylabel('真正率')
# plt.title('RF宏平均ROC曲线')
# plt.legend(loc="lower right")
# plt.grid(True)
#
# # 显示图形
# plt.show()
