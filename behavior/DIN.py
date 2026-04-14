import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import roc_auc_score,log_loss
from tqdm.auto import tqdm

# 读取
print("-----数据读取与拆分-----")
file_path = "../data/sample/DINSample.parquet"
data = pd.read_parquet(file_path)

data = data.sort_values('time_stamp').reset_index(drop=True) # 按时间升序

# 字段分组
target_cols = ['cate_id','brand']
hist_cols = {'cate_id': 'hist_cate_seq','brand': 'hist_brand_seq'} # 建立索引，后续共享Embedding
sparse_cols = ['adgroup_id', 'pid', 'cate_id', 'brand','cms_group_id', 'cms_segid', 'final_gender_code',
                'occupation', 'age_level', 'shopping_level','gender_cate_cross', 'hour', 'weekday']
dense_cols = ['price', 'ad_hist_imp', 'ad_hist_clk','user_hist_imp', 'user_hist_clk', 
              'user_hist_ctr','user_cate_hist_imp', 'user_cate_hist_clk', 'user_cate_hist_ctr']

# 时间切分
date_list = sorted(data['date'].astype(str).unique())
train_df = data[data['date'].astype(str).isin(date_list[:-2])].copy()
valid_df = data[data['date'].astype(str) == date_list[-2]].copy()
test_df = data[data['date'].astype(str) == date_list[-1]].copy()
print(f"时间范围:{data['date'].min()}~{data['date'].max()}\n训练集形状{train_df.shape}，验证集形状{valid_df.shape}，测试集形状{test_df.shape}\n")

max_len = int(data['seq_len'].max()) # 取当前样本的最大序列长度，其实就是100，便于后续padding

# 定义编码函数
print("-----稀疏特征编码 & 稠密特征归一化-----")
def build_vocab(values): #  建立词表，真实id从2开始编号，0留给padding，1留给OOV
    uniq_values = pd.Series(values).dropna().astype(str).unique().tolist()
    return {value: idx for idx,value in enumerate(uniq_values,start=2)}
def encode_scalar(series,vocab): # 编码标量列
    return series.astype(str).map(lambda x: vocab.get(x,1)).astype('int64')
def encode_seq(series,vocab): # 编码序列
    return series.map(lambda seq: np.array([vocab.get(str(x),1) for x in seq],dtype=np.int64))

# 共享特征编码：cate和brand
shared_vocabs = {}
for target_col,hist_col in hist_cols.items():
    # 将共享编码的两个特征合并建立词表
    vocab = build_vocab(pd.concat([train_df[target_col],train_df[hist_col].explode()]))
    shared_vocabs[target_col] = vocab
    # 用编码id代替原数值
    train_df[target_col] = encode_scalar(train_df[target_col], vocab)
    valid_df[target_col] = encode_scalar(valid_df[target_col], vocab)
    test_df[target_col] = encode_scalar(test_df[target_col], vocab)
    train_df[hist_col] = encode_seq(train_df[hist_col], vocab)
    valid_df[hist_col] = encode_seq(valid_df[hist_col], vocab)
    test_df[hist_col] = encode_seq(test_df[hist_col], vocab)
    print(f"{target_col}词表大小:{len(vocab) + 2}")

# sparse特征编码
for col in sparse_cols:
    if col in target_cols: continue
    vocab = build_vocab(train_df[col])
    train_df[col] = encode_scalar(train_df[col], vocab)
    valid_df[col] = encode_scalar(valid_df[col], vocab)
    test_df[col] = encode_scalar(test_df[col], vocab)
    print(f"{col}词表大小:{len(vocab) + 2}")
    
# 稠密特征标准化
mean = train_df[dense_cols].mean()
std = train_df[dense_cols].std().replace(0,1)
train_df[dense_cols] = (train_df[dense_cols] - mean) / std
valid_df[dense_cols] = (valid_df[dense_cols] - mean) / std
test_df[dense_cols] = (test_df[dense_cols] - mean) / std

vocab_sizes = {}
for col in sparse_cols: # 统计每个sparse字段的最大编号，用来确定所建Embedding表的大小
    if col in target_cols: continue
    max_idx = max(train_df[col].max(),valid_df[col].max(),test_df[col].max())
    vocab_sizes[col] = int(max_idx) + 1
for target_col in target_cols:
    vocab_sizes[target_col] = len(shared_vocabs[target_col]) + 2

# 开始构建DIN
class DIN(nn.Module):
    def __init__(self,vocab_sizes,emb_dim=8):
        super().__init__()
        self.emb_dim = emb_dim
        # self.embedding用来存储每个特征的向量表
        self.embedding = nn.ModuleDict()
        for col,size in vocab_sizes.items():
            self.embedding[col] = nn.Embedding(size,emb_dim,padding_idx=0)
        # Attention MLP
        self.attn_fc1 = nn.Linear(emb_dim*4,32)
        self.attn_fc2 = nn.Linear(32,1)
        self.relu = nn.ReLU()
        # DNN
        # fc1总维度的来源：[len(sparse_cols)-len(target_cols)] + [len(target_cols)] + [len(target_cols)] + [len(dense_cols)]
        self.fc1 = nn.Linear((len(sparse_cols)+len(target_cols)) * emb_dim + len(dense_cols),128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,1)
        self.dropout = nn.Dropout(0.2)
    
    def activateInterest(self,target_emb,hist_emb,seq_len):
        ''' DIN最核心的创新：构造用户兴趣向量'''
        max_len = hist_emb.size(1) # 取历史序列长度L
        target_emb = target_emb.unsqueeze(1).expand(-1,max_len,-1) # 将target_emb从[B,E]变为[B,L,E]，每一个L的位置其实还是同一个向量
        attn_input = torch.cat([target_emb,hist_emb,target_emb-hist_emb,target_emb*hist_emb],dim=-1) # 显式构建差异和交叉
        # 根据历史序列和当前商品打分
        score = self.attn_fc2(self.relu(self.attn_fc1(attn_input))).squeeze(-1)
        # 序列mask，超出实际序列长度的部分置为false
        mask = torch.arange(max_len,device=seq_len.device).unsqueeze(0) < seq_len.unsqueeze(1)
        score = score.masked_fill(~mask,-1e9) # 把padding位置替换为极小值，后续softmax后便可当成0
        weight = torch.softmax(score,dim=1)
        interest = torch.sum(hist_emb*weight.unsqueeze(-1),dim=1)
        return interest
    
    def forward(self,x): 
        sparse_embs = []
        for col in sparse_cols:
            if col in target_cols: continue # cate_id和brand要单独作为target ad特征处理
            sparse_embs.append(self.embedding[col](x[col]))
        target_embs = {}
        for col in target_cols:
            target_embs[col] = self.embedding[col](x[col])
        interest_embs = {}
        for target_col,hist_col in hist_cols.items():
            interest_embs[target_col] = self.activateInterest(
                target_embs[target_col],
                self.embedding[target_col](x[hist_col]),
                x['seq_len']
            )
        # 合并每个特征向量
        sparse_stack = torch.cat(sparse_embs,dim=1) # dim=1的意思是按照第一维开始拼接，跳过Batch这个第零维
        target_stack = torch.cat([target_embs[col] for col in target_cols],dim=1)
        interest_stack = torch.cat([interest_embs[col] for col in target_cols],dim=1)
        dense_stack = torch.stack([x[col] for col in dense_cols],dim=1).float()
        dnn_input = torch.cat([sparse_stack,target_stack,interest_stack,dense_stack],dim=1)
        hidden = self.dropout(self.relu(self.fc1(dnn_input)))
        hidden = self.dropout(self.relu(self.fc2(hidden)))
        logit = self.fc3(hidden).squeeze(-1)
        return logit
    
class DINDataset(Dataset):
    def __init__(self,df,max_len):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
    
    def padding(self,seq): # 长度不足的直接补0
        seq = np.asarray(seq,dtype=np.int64)
        width = self.max_len - len(seq)
        return np.pad(seq,(0,width),mode='constant',constant_values=0)
    
    def __getitem__(self,index):
        row = self.df.iloc[index]
        y = torch.tensor(row['clk'],dtype=torch.float32)
        x = {}
        for col in sparse_cols:
            x[col] = torch.tensor(row[col],dtype=torch.long)
        for col in hist_cols.values():
            x[col] = torch.tensor(self.padding(row[col]),dtype=torch.long)
        x['seq_len'] = torch.tensor(row['seq_len'],dtype=torch.long)
        for col in dense_cols:
            x[col] = torch.tensor(row[col],dtype=torch.float32)
        return x,y
            
train_dataset = DINDataset(train_df,max_len)
valid_dataset = DINDataset(valid_df,max_len)
test_dataset = DINDataset(test_df,max_len)

train_loader = DataLoader(train_dataset,batch_size=256,shuffle=True)
valid_loader = DataLoader(valid_dataset,batch_size=256,shuffle=False)
test_loader = DataLoader(test_dataset,batch_size=256,shuffle=False)

SEED = 42
torch.manual_seed(SEED)

model = DIN(
    vocab_sizes=vocab_sizes,
    emb_dim=8
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def evaluate(model,loader,criterion,device):
    model.eval()
    total_loss = 0
    y_true,y_pred = [],[]
    with torch.no_grad():
        for x,y in loader:
            x = {k:v.to(device) for k,v in x.items()}
            y = y.to(device)
            logit = model(x)
            loss = criterion(logit,y) # loss默认为平均值，所以计算total时要乘样本数
            total_loss += loss.item()*y.size(0)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(torch.sigmoid(logit).cpu().numpy().tolist())
    total_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(y_true,y_pred)
    logloss = log_loss(y_true,y_pred)
    return total_loss,auc,logloss

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=3e-4) # type: ignore
epochs = 8
best_val_auc = 0
best_state = None
best_epoch = 0

print("\n-----开始训练模型-----")
for epoch in range(1,epochs+1):
    model.train()
    train_loss_sum = 0
    pbar = tqdm(train_loader,desc=f"第{epoch}/{epochs}轮",leave=False)   
    for x,y in pbar:
        x = {k: v.to(device) for k, v in x.items()}
        y = y.to(device)
        optimizer.zero_grad()
        logit = model(x)
        loss = criterion(logit,y)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item() * y.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    train_loss = train_loss_sum / len(train_loader.dataset) # type: ignore
    val_loss,val_auc,val_logloss = evaluate(model,valid_loader,criterion,device)
    print(
        f"第{epoch}/{epochs}轮 | "
        f"训练集损失={train_loss:.4f} | "
        f"验证集损失={val_loss:.4f} | "
        f"验证集auc={val_auc:.4f} | "
        f"验证集logloss={val_logloss:.4f}"
    )
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_epoch = epoch
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
if best_state is not None:
    print(f"\n最佳模型来自第{best_epoch}轮，验证集AUC={best_val_auc:.4f}")
    model.load_state_dict(best_state)

test_loss,test_auc,test_logloss = evaluate(model,test_loader,criterion,device)
print(
    f"测试集：loss={test_loss:.4f} | "
    f"auc={test_auc:.4f} | "
    f"logloss={test_logloss:.4f}"
)