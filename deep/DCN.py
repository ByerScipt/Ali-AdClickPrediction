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
sparse_cols = ['pid','hour','weekday','cate_id','brand','cms_group_id','cms_segid','final_gender_code','occupation']
dense_cols = ['price','age_level','shopping_level','ad_hist_imp','ad_hist_clk','user_hist_imp','user_hist_clk',
                   'user_hist_ctr','user_cate_hist_imp','user_cate_hist_clk','user_cate_hist_ctr']

# 切分训练集和验证集，而后删去date
last_date = data['date'].max()
train_data = data[data['date']!=last_date].copy()
valid_data = data[data['date']==last_date].copy()
train_data = train_data.drop(columns=['date'])
valid_data = valid_data.drop(columns=['date'])

# 稠密数据标准化
mean = train_data[dense_cols].mean()
std = train_data[dense_cols].std().replace(0,1)
train_data[dense_cols] = (train_data[dense_cols] - mean) / std
valid_data[dense_cols] = (valid_data[dense_cols] - mean) / std

# 将数据结构转为torch张量
train_sparse = torch.tensor(train_data[sparse_cols].to_numpy(),dtype=torch.long)
train_dense = torch.tensor(train_data[dense_cols].to_numpy(),dtype=torch.float32)
train_y = torch.tensor(train_data['clk'].to_numpy(),dtype=torch.float32)

valid_sparse = torch.tensor(valid_data[sparse_cols].to_numpy(),dtype=torch.long)
valid_dense = torch.tensor(valid_data[dense_cols].to_numpy(),dtype=torch.float32)
valid_y = torch.tensor(valid_data['clk'].to_numpy(),dtype=torch.float32)

# 构建Dataset
train_dataset = TensorDataset(train_sparse,train_dense,train_y)
valid_dataset = TensorDataset(valid_sparse,valid_dense,valid_y)

# 构建DataLoader
batch_size = 1024
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
valid_loader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False)
print(f"处理完成!")

# 开始搭建模型
SEED = 42
sparse_sizes = {col:int(data[col].max())+1 for col in sparse_cols}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)

class CrossLayer(nn.Module):
    def __init__(self,input_dim,rank):
        super().__init__()
        self.v = nn.Linear(input_dim,rank,bias=False)
        self.u = nn.Linear(rank,input_dim,bias=True)
        
    def forward(self,x0,xl):
        # x_{L+1} = x0 ⊙ (W_l*x_L + b_L) + x_L
        return x0 * self.u(self.v(xl)) + xl 

class DCN(nn.Module):
    def __init__(self,sparse_sizes,dense_dim,emb_dim=8,num_cross_layers=2,hidden_dims=(64,32),dropout=0.2,rank=32):
        super().__init__()
        self.sparse_emb = nn.ModuleDict({col: nn.Embedding(size,emb_dim) for col,size in sparse_sizes.items()})
        self.input_dim = len(sparse_sizes) * emb_dim + dense_dim
        self.cross_layers = nn.ModuleList([CrossLayer(self.input_dim,rank) for _ in range(num_cross_layers)])
        
        # 开始构建deep network
        deep_layers = []
        cur_dim = self.input_dim
        for hdim in hidden_dims:
            deep_layers.append(nn.Linear(cur_dim,hdim))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(dropout))
            cur_dim = hdim
        self.deep_net = nn.Sequential(*deep_layers)
        self.output_layer = nn.Linear(cur_dim,1)
        
    def forward(self,sparse,dense):
        emb_parts = []
        for i,col in enumerate(sparse_cols):
            emb = self.sparse_emb[col](sparse[:,i])
            emb_parts.append(emb)
        x0 = torch.cat(emb_parts+[dense],dim=1)
        x_cross = x0
        for cross_layer in self.cross_layers:
            x_cross = cross_layer(x0,x_cross)
        deep_out = self.deep_net(x_cross)
        logit = self.output_layer(deep_out)
        return logit.squeeze(1)
    
model = DCN(
    sparse_sizes=sparse_sizes,
    dense_dim=len(dense_cols),
    emb_dim=8,
    num_cross_layers=2,
    hidden_dims=(64,32),
    dropout=0.2,
    rank=64
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5) # type: ignore

print("--------开始训练模型--------")
epochs = 25
best_auc = 0
best_logloss = float("inf")
patience = 5
wait = 0

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for sparse,dense,y in train_loader:
        sparse = sparse.to(device)
        dense = dense.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        logits = model(sparse,dense)
        loss = criterion(logits,y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5)
        optimizer.step()
        
        train_loss += loss.item() * y.size(0)
    train_loss = train_loss / len(train_dataset)
    
    model.eval()
    valid_loss = 0
    preds,labels = [],[]
    with torch.no_grad():
        for sparse,dense,y in valid_loader:
            sparse = sparse.to(device)
            dense = dense.to(device)
            y = y.to(device)
            logits = model(sparse,dense)
            loss = criterion(logits,y)
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
    if auc > best_auc + 1e-5:
        best_auc = auc
        best_logloss = logloss
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"验证AUC连续{patience}轮未提升，提前停止训练")
            break
print(f"最佳AUC:{best_auc:.6f}, 对应LogLoss:{best_logloss:.6f}")