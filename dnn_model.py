import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, \
    confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
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
X_train, X_test, y_train, y_test = train_test_split(X_processed, targets, test_size=0.3, random_state=42)

# 将数据转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)


# 定义DNN模型
class DNNClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DNNClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 12)
        self.layer3 = nn.Linear(12, 6)
        self.output_layer = nn.Linear(6, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.output_layer(x)
        return x


input_dim = X_train.shape[1]
output_dim = y_train.shape[1]  # 输出的维度等于标签的数量
model = DNNClassifier(input_dim, output_dim)
criterion = nn.BCEWithLogitsLoss()  # 使用二分类交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 使用Adam优化器


# 训练模型函数
def train_model(num_epochs):
    model.train()  # 设置模型为训练模式
    loss_history = []
    acc_history = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()  # 梯度清零
        outputs = model(X_train)  # 正向传播
        loss = criterion(outputs, y_train)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

        # 计算训练准确率
        predicted = torch.sigmoid(outputs)
        predicted_binary = (predicted > 0.5).float()
        accuracy = (predicted_binary == y_train).float().mean().item()

        loss_history.append(loss.item())
        acc_history.append(accuracy)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

    # 绘制训练损失图和准确率图
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Loss')
    plt.title('训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(acc_history, label='Accuracy')
    plt.title('训练准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig('dnn_training_metrics.png')
    plt.close()


# 训练模型
train_model(500)
print(type(y_test))
y_test = pd.DataFrame(y_test.numpy())
# 初始化变量来存储总的评估指标
total_accuracy = 0
total_precision = 0
total_recall = 0
total_f1 = 0
total_auc = 0
num_labels = len(targets.columns)

# 评估模型
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    outputs = model(X_test)
    predicted = torch.sigmoid(outputs)  # 使用sigmoid函数将输出值转换为概率

    predicted_binary = (predicted > 0.5).float()  # 将概率二值化

    # # 对每个标签分别计算评估指标
    # for i, label in enumerate(targets.columns):
    #     accuracy = accuracy_score(y_test[:, i], predicted_binary[:, i])
    #     precision = precision_score(y_test[:, i], predicted_binary[:, i])
    #     recall = recall_score(y_test[:, i], predicted_binary[:, i])
    #     f1 = f1_score(y_test[:, i], predicted_binary[:, i])
    #     auc = roc_auc_score(y_test[:, i], predicted[:, i])
    #
    #     print(f'{label} - 准确率: {accuracy:.2f}, 精确率: {precision:.2f}, 召回率: {recall:.2f}, '
    #           f'F1得分: {f1:.2f}, AUC: {auc:.2f}')
    #
    #     # 累积总的评估指标
    #     total_accuracy += accuracy
    #     total_precision += precision
    #     total_recall += recall
    #     total_f1 += f1
    #     total_auc += auc
#
#         # 绘制ROC曲线
#         fpr, tpr, _ = roc_curve(y_test[:, i], predicted[:, i])
#         plt.figure()
#         plt.plot(fpr, tpr, label='AUC = %0.2f' % auc)
#         plt.plot([0, 1], [0, 1], 'r--')
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.05])
#         plt.xlabel('假阳性率')
#         plt.ylabel('真正率')
#         plt.title(f'{label} - ROC曲线')
#         plt.legend(loc="lower right")
#         plt.savefig(f'dnn_roc_curve_{label}.png')
#         plt.close()
#
#         # 绘制混淆矩阵
#         cm = confusion_matrix(y_test[:, i], predicted_binary[:, i])
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
#                     yticklabels=['Negative', 'Positive'])
#         plt.xlabel('预测标签')
#         plt.ylabel('真实标签')
#         plt.title(f'{label} - 混淆矩阵')
#         plt.savefig(f'dnn_confusion_matrix_{label}.png')
#         plt.close()
#
#
# 计算平均指标
# average_accuracy = total_accuracy / num_labels
# average_precision = total_precision / num_labels
# average_recall = total_recall / num_labels
# average_f1 = total_f1 / num_labels
# average_auc = total_auc / num_labels
#
# # 输出平均指标
# print(f'\n平均指标 - 准确率: {average_accuracy:.2f}, 精确率: {average_precision:.2f}, 召回率: {average_recall:.2f}, '
#       f'F1得分: {average_f1:.2f}, AUC: {average_auc:.2f}')

# 初始化用于计算平均ROC曲线的数据
all_fpr = np.linspace(0, 1, 100)
mean_tpr = 0.0

# 对每个标签分别计算ROC曲线并累计TPR
for i, label in enumerate(targets.columns):
    fpr, tpr, _ = roc_curve(y_test.iloc[:, i], predicted[:, i].numpy())  # 使用 predicted[:, i] 获取每个类的预测值
    mean_tpr += np.interp(all_fpr, fpr, tpr)  # 插值计算TPR
    mean_tpr[0] = 0.0  # 确保第一个点是(0,0)


# 计算平均TPR
mean_tpr /= num_labels
mean_tpr[-1] = 1.0  # 确保最后一个点是(1,1)

# 计算宏平均AUC
mean_auc = auc(all_fpr, mean_tpr)
# 确保将 fpr 和 tpr 单独存储为列，而不是将它们打包成列表
roc_data = pd.DataFrame({
    'fpr': all_fpr,  # 直接使用 all_fpr 数组
    'tpr': mean_tpr  # 使用计算后的 mean_tpr 数组
})

# # 单独保存 AUC 值（如需要保存为 CSV 文件中的附加信息）
# auc_value = pd.DataFrame({'auc': [mean_auc]})
roc_data['auc'] = [mean_auc] * len(roc_data)
# 保存到文件
roc_data.to_csv('dnn_roc_data.csv', index=False)
# auc_value.to_csv('dnn_auc_value.csv', index=False)
# 绘制宏平均ROC曲线
plt.figure(figsize=(10, 8))
plt.plot(all_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.3f})', lw=2)

# 绘制对角线
plt.plot([0, 1], [0, 1], 'r--')

# 设置图形标题和标签
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('DNN平均ROC曲线')
plt.legend(loc="lower right")
plt.grid(True)

# 显示图形
plt.show()