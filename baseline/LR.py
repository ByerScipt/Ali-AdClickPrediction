'''
LR难以处理高基类数据，所以只使用以下9个特征
类别特征(one-hot)：hour, weekday, pid, cate_id, cms_group_id, final_gender_code, occupation
等级特征(scaling)：age_level, shopping_level
'''
import pandas as pd

# 读取
print("数据读取与处理...")
file_path = "../data/processed_data/LRSample.csv"
data = pd.read_csv(file_path)

# 提取时间信息
time_stamp = pd.to_datetime(data['time_stamp'],unit='s')
data['hour'] = time_stamp.dt.hour
data['weekday'] = time_stamp.dt.weekday
data['date'] = time_stamp.dt.date
last_date = data['date'].max()

# 删除无用列
data = data.drop(columns=['pvalue_level','new_user_class_level','brand','cms_segid','user','time_stamp','customer','campaign_id','adgroup_id','price'])

# 切分训练集和验证集，而后删去date
train_data = data[data['date']!=last_date].copy()
valid_data = data[data['date']==last_date].copy()
train_data = train_data.drop(columns=['date'])
valid_data = valid_data.drop(columns=['date'])
# print(f"训练集形状{train_data.shape}，平均点击率{train_data['clk'].mean()}")
# print(f"验证集形状{valid_data.shape}，平均点击率{valid_data['clk'].mean()}")

# 进一步切分类别、等级和标签
cate_cols = ['hour','weekday','pid','cate_id','cms_group_id','final_gender_code','occupation']
num_cols = ['age_level','shopping_level']
feat_cols = cate_cols + num_cols
x_train = train_data[feat_cols]
y_train = train_data['clk']
x_valid = valid_data[feat_cols]
y_valid = valid_data['clk']
# print(f"\n训练集特征形状{x_train.shape},标签形状{y_train.shape}")
# print(f"验证集特征形状{x_valid.shape},标签形状{y_valid.shape}")

# one-hot
x_train = pd.get_dummies(x_train,columns=cate_cols)
x_valid = pd.get_dummies(x_valid,columns=cate_cols)
x_valid = x_valid.reindex(columns=x_train.columns,fill_value=0) # 对齐验证集与训练集

# scaling
mean = x_train[num_cols].mean()
std = x_train[num_cols].std().replace(0,1) # 避免除0
x_train[num_cols] = (x_train[num_cols] - mean) / std
x_valid[num_cols] = (x_valid[num_cols] - mean) / std
# print(x_train[num_cols].head())
# print(x_valid[num_cols].head())
print("处理完成!")
print(f"当前训练集形状{x_train.shape},验证集形状{x_valid.shape}\n")

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

print("开始训练逻辑回归模型...")
max_iter = 100
model = LogisticRegression(random_state=13, max_iter=max_iter)
model.fit(x_train,y_train)
print(f"模型训练完成，训练轮数{max_iter}轮")

y_predict = model.predict_proba(x_valid)
score = roc_auc_score(y_true=y_valid,y_score=y_predict[:,1])
loss = log_loss(y_true=y_valid,y_pred=y_predict)

print(f"验证集AUC:{score}\n验证集logloss:{loss}")
