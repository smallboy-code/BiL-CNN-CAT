import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据集
data = pd.read_excel('../500例IgA数据集.xlsx', sheet_name='data')
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
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

X_processed = preprocessor.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(X_processed, targets, test_size=0.3, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# 定义带有注意力机制的LSTM模型
class AttentionLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(AttentionLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)  # 注意力层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # 计算注意力权重
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)  # 加权求和得到上下文向量
        output = self.fc(context_vector)
        return output

input_dim = X_train.shape[2]
hidden_dim = 32
output_dim = y_train.shape[1]
model = AttentionLSTMClassifier(input_dim, hidden_dim, output_dim)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练模型函数
def train_model(num_epochs):
    model.train()
    loss_history = []
    acc_history = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        predicted = torch.sigmoid(outputs)
        predicted_binary = (predicted > 0.5).float()
        accuracy = (predicted_binary == y_train).float().mean().item()

        loss_history.append(loss.item())
        acc_history.append(accuracy)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

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

    plt.savefig('attention_lstm_training_metrics.png')
    plt.close()

train_model(500)

# 评估模型
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    predicted = torch.sigmoid(outputs)
    predicted_binary = (predicted > 0.5).float()

    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_auc = 0
    num_labels = len(targets.columns)

    for i, label in enumerate(targets.columns):
        accuracy = accuracy_score(y_test[:, i], predicted_binary[:, i])
        precision = precision_score(y_test[:, i], predicted_binary[:, i])
        recall = recall_score(y_test[:, i], predicted_binary[:, i])
        f1 = f1_score(y_test[:, i], predicted_binary[:, i])
        auc = roc_auc_score(y_test[:, i], predicted[:, i])

        print(f'{label} - 准确率: {accuracy:.2f}, 精确率: {precision:.2f}, 召回率: {recall:.2f}, '
              f'F1得分: {f1:.2f}, AUC: {auc:.2f}')

        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_auc += auc

        fpr, tpr, _ = roc_curve(y_test[:, i], predicted[:, i])
        plt.figure()
        plt.plot(fpr, tpr, label='AUC = %0.2f' % auc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真正率')
        plt.title(f'{label} - ROC曲线')
        plt.legend(loc="lower right")
        plt.savefig(f'attention_lstm_roc_curve_{label}.png')
        plt.close()

        cm = confusion_matrix(y_test[:, i], predicted_binary[:, i])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title(f'{label} - 混淆矩阵')
        plt.savefig(f'attention_lstm_confusion_matrix_{label}.png')
        plt.close()

    average_accuracy = total_accuracy / num_labels
    average_precision = total_precision / num_labels
    average_recall = total_recall / num_labels
    average_f1 = total_f1 / num_labels
    average_auc = total_auc / num_labels

    print(f'\n平均指标 - 准确率: {average_accuracy:.2f}, 精确率: {average_precision:.2f}, 召回率: {average_recall:.2f}, '
          f'F1得分: {average_f1:.2f}, AUC: {average_auc:.2f}')
