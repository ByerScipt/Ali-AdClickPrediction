import torch.nn as nn
import torch
from torch.utils.data import TensorDataset,DataLoader
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss

# 读取
print("数据读取与处理...")
file_path = "../data/processed_data/WideDeepSample.parquet"
data = pd.read_parquet(file_path)

# 拆分稀疏与稠密特征
wide_sparse_cols = ['gender_cate_cross','pid','hour','weekday','cate_id','brand']
wide_dense_cols = ['price','ad_hist_imp','ad_hist_clk','user_hist_ctr','user_cate_hist_ctr']
deep_sparse_cols = ['pid','cate_id','brand','cms_group_id','cms_segid',
                    'final_gender_code','occupation','hour','weekday']
deep_dense_cols = ['price','age_level','shopping_level','ad_hist_imp','ad_hist_clk','user_hist_imp','user_hist_clk',
                   'user_hist_ctr','user_cate_hist_imp','user_cate_hist_clk','user_cate_hist_ctr']

# 切分训练集和验证集，而后删去date
last_date = data['date'].max()
train_data = data[data['date']!=last_date].copy()
valid_data = data[data['date']==last_date].copy()
train_data = train_data.drop(columns=['date'])
valid_data = valid_data.drop(columns=['date'])

# 稠密数据标准化
dense_cols = list(set(wide_dense_cols+deep_dense_cols))
mean = train_data[dense_cols].mean()
std = train_data[dense_cols].std().replace(0,1)
train_data[dense_cols] = (train_data[dense_cols] - mean) / std
valid_data[dense_cols] = (valid_data[dense_cols] - mean) / std

# 将数据结构转为torch张量
train_wide_sparse_x = torch.tensor(train_data[wide_sparse_cols].to_numpy(),dtype=torch.long)
train_wide_dense_x = torch.tensor(train_data[wide_dense_cols].to_numpy(),dtype=torch.float32)
train_deep_sparse_x = torch.tensor(train_data[deep_sparse_cols].to_numpy(),dtype=torch.long)
train_deep_dense_x = torch.tensor(train_data[deep_dense_cols].to_numpy(),dtype=torch.float32)
train_y = torch.tensor(train_data['clk'].to_numpy(),dtype=torch.float32)

valid_wide_sparse_x = torch.tensor(valid_data[wide_sparse_cols].to_numpy(),dtype=torch.long)
valid_wide_dense_x = torch.tensor(valid_data[wide_dense_cols].to_numpy(),dtype=torch.float32)
valid_deep_sparse_x = torch.tensor(valid_data[deep_sparse_cols].to_numpy(),dtype=torch.long)
valid_deep_dense_x = torch.tensor(valid_data[deep_dense_cols].to_numpy(),dtype=torch.float32)
valid_y = torch.tensor(valid_data['clk'].to_numpy(),dtype=torch.float32)

# 构建Dataset
train_dataset = TensorDataset(
    train_wide_sparse_x, train_wide_dense_x,
    train_deep_sparse_x, train_deep_dense_x,
    train_y
)
valid_dataset = TensorDataset(
    valid_wide_sparse_x, valid_wide_dense_x,
    valid_deep_sparse_x, valid_deep_dense_x,
    valid_y
)

# 构建DataLoader
batch_size = 1024
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
valid_loader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False)
print(f"处理完成!")

# 开始搭建模型
wide_sparse_sizes = {col:int(data[col].max())+1 for col in wide_sparse_cols}
deep_sparse_sizes = {col:int(data[col].max())+1 for col in deep_sparse_cols}

class WideDeep(nn.Module):
    def __init__(self,wide_sparse_sizes,deep_sparse_sizes,wide_dense_dim,deep_dense_dim,emb_dim=8):
        super().__init__()  # PyTorch自定义类的固定写法
        
        # 定义Wide侧，维度均为1，即每个类别只需要学习一个权重
        self.wide_sparse_emb = nn.ModuleDict({
            col:nn.Embedding(size,1) for col,size in wide_sparse_sizes.items()
        })
        self.wide_dense_linear = nn.Linear(wide_dense_dim,1)
        
        # 定义Deep侧，稀疏特征的维度暂定为8，而后与稠密特征拼接
        self.deep_sparse_emb = nn.ModuleDict({
            col:nn.Embedding(size,emb_dim) for col,size in deep_sparse_sizes.items()
        })
        deep_input_dim = len(deep_sparse_cols)*emb_dim + deep_dense_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(deep_input_dim,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32,1)
        )
        
    def forward(self,wide_sparse,wide_dense,deep_sparse,deep_dense):
        # 计算Wide侧logit
        wide_sparse_parts = []
        for i,col in enumerate(wide_sparse_cols):
            emb = self.wide_sparse_emb[col](wide_sparse[:,i])
            wide_sparse_parts.append(emb)
        wide_sparse_parts = torch.cat(wide_sparse_parts,dim=1)
        wide_logit = wide_sparse_parts.sum(dim=1,keepdim=True) + self.wide_dense_linear(wide_dense)
        
        # Deep侧Logit
        deep_sparse_parts = []
        for i,col in enumerate(deep_sparse_cols):
            emb = self.deep_sparse_emb[col](deep_sparse[:,i])
            deep_sparse_parts.append(emb)
        deep_sparse_parts = torch.cat(deep_sparse_parts,dim=1)
        deep_input = torch.cat([deep_sparse_parts,deep_dense],dim=1)
        deep_logit = self.mlp(deep_input)
        
        logit = wide_logit + deep_logit
        return logit.squeeze(1)
        
model = WideDeep(
    wide_sparse_sizes=wide_sparse_sizes,
    deep_sparse_sizes=deep_sparse_sizes,
    wide_dense_dim=len(wide_dense_cols),
    deep_dense_dim=len(deep_dense_cols),
    emb_dim=8
)

Loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3) # type: ignore

# 训练部分
print("--------开始训练模型--------")
epochs = 25
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for wide_sparse,wide_dense,deep_sparse,deep_dense,y in train_loader:
        optimizer.zero_grad()  # 清理上一批残留的梯度
        logits = model(wide_sparse,wide_dense,deep_sparse,deep_dense)
        loss = Loss(logits,y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * y.size(0)
    train_loss = train_loss / len(train_dataset)
    
    model.eval()
    valid_loss = 0
    preds,labels = [],[]
    with torch.no_grad():
        for wide_sparse,wide_dense,deep_sparse,deep_dense,y in valid_loader:
            logits = model(wide_sparse,wide_dense,deep_sparse,deep_dense)
            loss = Loss(logits,y)
            pred = torch.sigmoid(logits)
            valid_loss += loss.item() * y.size(0)
            preds.extend(pred.cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())
    valid_loss = valid_loss / len(valid_dataset)
    auc = roc_auc_score(labels,preds)
    logloss = log_loss(labels,preds)
    
    print(f"第{epoch+1}/{epochs}轮")
    print(f"训练集损失:{train_loss:.6f},验证集损失:{valid_loss:.6f}")
    print(f"验证集AUC:{auc:.6f},LogLoss:{logloss:.6f}\n")