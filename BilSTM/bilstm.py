import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from transformers import BertConfig
from transformers import BertConfig, BertModel
device = torch.device("cpu")
print(f"Using device: {device}")
# ------------------ 数据预处理 ------------------ #
file_path = "数据集.xlsx"  # 替换为你的文件路径
df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()

# 数值型特征标准化
numeric_features = ["年龄", "尿蛋白定量", "收缩压", "舒张压", "肾小球滤过率（MDRD）"]

# 将非数值数据转换为 NaN
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 使用均值填充缺失值
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())

# 标准化数值特征
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])
df[numeric_features] = df[numeric_features].astype("float32")

# 分类型特征编码
categorical_features = ["头晕，血压高", "肾功能重度下降(GFR＜30）", "口气重，呼气时有尿臭",
                        "疲惫/困乏/乏力", "易感冒", "自汗/盗汗", "恶风", "皮肤瘙痒", "肌肉/头身/肢节酸楚",
                        "水肿加重", "腰酸", "气短懒言", "夜尿增多（夜间睡眠时尿量＞750ml或大于白天尿量）",
                        "手足心热", "目睛干涩", "咽干咽燥", "腰痛固定", "久病(病程≥3个月）",
                        "皮肤瘀斑、瘀点/肌肤甲错/皮肤赤丝红缕/蟹爪纹络", "肢体麻木", "面色黧黑", "急躁易怒",
                        "头痛", "视物模糊甚则黑蒙", "震颤，搐搦", "纳呆、泛恶", "面色不华", "畏寒怕冷"]
df[categorical_features] = df[categorical_features].replace({"是": 1, "否": 0})
print(df[categorical_features].dtypes)
print(df[categorical_features].head())

# 合并文本特征
df["文本特征"] = df["舌质"] + " " + df["舌苔"]
df["舌质"] = df["舌质"].fillna("")
df["舌苔"] = df["舌苔"].fillna("")
df["文本特征"] = df["舌质"] + " " + df["舌苔"]

# 标签处理
label_columns = ["肾虚（肾气阴）证", "风湿证", "瘀痹证", "肝风证", "溺毒证"]
df[label_columns] = df[label_columns].astype(float)

# 划分训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ------------------ 数据集类 ------------------ #
class CustomDataset(Dataset):
    def __init__(self, data, numeric_features, categorical_features, tokenizer, max_len, label_columns):
        self.data = data
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_columns = label_columns

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 检查并转换 numeric_features
        numeric_features = torch.tensor(
                [float(x) for x in row[self.numeric_features].values],
                dtype=torch.float32
            )
        categorical_features = torch.tensor(
            [int(x) for x in row[self.categorical_features].values],
            dtype=torch.int64  # 确保分类型特征为 int64
        )

        inputs = self.tokenizer(
            row["文本特征"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        token_type_ids = inputs["token_type_ids"].squeeze()
        numeric_features = torch.tensor(row[self.numeric_features].values, dtype=torch.float32)  # 数值特征为 float32
        categorical_features = torch.tensor(row[self.categorical_features].values, dtype=torch.int64)  # 分类特征为 int64

        labels = torch.tensor(row[self.label_columns].values, dtype=torch.float)
        return numeric_features, categorical_features, input_ids, attention_mask, token_type_ids, labels


# ------------------ 数据加载器 ------------------ #
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
train_dataset = CustomDataset(train_df, numeric_features, categorical_features, tokenizer, 128, label_columns)
test_dataset = CustomDataset(test_df, numeric_features, categorical_features, tokenizer, 128, label_columns)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
for numeric, categorical, input_ids, attention_mask, token_type_ids, labels in train_loader:
    print(f"Numeric features shape: {numeric.shape}, dtype: {numeric.dtype}")
    print(f"Categorical features shape: {categorical.shape}, dtype: {categorical.dtype}")
    print(f"Input IDs shape: {input_ids.shape}, dtype: {input_ids.dtype}")
    print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
    break

checkpoint = "bert-base-chinese"
# ------------------ 模型定义 ------------------ #
class LSTM(nn.Module):
    def __init__(self, outdim):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=21, embedding_dim=768)
        self.lstm = nn.LSTM(input_size=768, hidden_size=outdim, num_layers=2, batch_first=True, bidirectional=True)
        self.fcn = nn.Sequential(nn.Linear(66 * outdim * 2, outdim), nn.GELU())

    def forward(self, input):
        input = self.embedding(input)
        output, (_, _) = self.lstm(input)
        output = output.flatten(1, -1)
        output = self.fcn(output)
        return output


class BERT(nn.Module):
    def __init__(self, outdim, checkpoint):
        super(BERT, self).__init__()
        config = BertConfig.from_pretrained(checkpoint)
        self.bert = BertModel.from_pretrained(checkpoint, config=config)
        self.fcn = nn.Sequential(nn.Linear(768, outdim), nn.GELU())
        self.attn = nn.MultiheadAttention(embed_dim=outdim, num_heads=1, batch_first=True)

    def forward(self, input_ids, token_type_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).pooler_output
        output = self.fcn(output)
        input = output
        output, _ = self.attn(input, input, input)
        return output + input


class Model(nn.Module):
    def __init__(self, device, checkpoint, LSTMdim=80, BERTdim=80):
        super(Model, self).__init__()
        self.device = device
        self.bilstm = LSTM(LSTMdim)  # 初始化 BiLSTM
        self.bert = BERT(BERTdim, checkpoint)  # 初始化 BERT
        Catdim = LSTMdim + BERTdim  # 拼接维度
        self.fc = nn.Sequential(nn.Linear(Catdim, 4), nn.Sigmoid())  # 定义全连接层

    def forward(self, numeric_input, categorical_input, input_ids, token_type_ids, attention_mask):
        # BiLSTM 输出
        lstm_output = self.bilstm(categorical_input)  # LSTM 使用分类输入
        # BERT 输出
        bert_output = self.bert(input_ids, token_type_ids, attention_mask)
        # 拼接 BiLSTM 和 BERT 的输出
        combined_output = torch.cat([lstm_output, bert_output], dim=1)
        print(f"LSTM output shape: {lstm_output.shape}")
        print(f"BERT output shape: {bert_output.shape}")

        # 输出 logits
        logits = self.fc(combined_output)
        return logits



# ------------------ 训练与评估 ------------------ #
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for numeric, categorical, input_ids, attention_mask, token_type_ids, labels in loader:
        numeric, categorical = numeric.to(device), categorical.to(device)
        input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(numeric, categorical, input_ids, token_type_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for numeric, categorical, input_ids, attention_mask, token_type_ids, labels in loader:
            numeric, categorical = numeric.to(device), categorical.to(device)
            input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device)
            labels = labels.to(device)

            outputs = model(numeric, categorical, input_ids, token_type_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(loader)

# 初始化模型
model = Model(device=device, checkpoint=checkpoint).to(device)

# 损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 训练模型
epochs = 10
for epoch in range(epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss = evaluate(model, test_loader, criterion, device)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
