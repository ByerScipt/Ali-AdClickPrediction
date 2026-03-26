'''
强baseline使用LGBM，原因是LGBM可以建模非线性关系，学习特征交互，且对数据预处理要求没有LR这么严格
为了控制变量，暂时也只用LR使用的9个特征
'''
import pandas as pd

# 读取
print("数据读取与处理...")
file_path = "../data/processed_data/baselineSample.parquet"
data = pd.read_parquet(file_path)

# 切分训练集和验证集，而后删去date
last_date = data['date'].max()
train_data = data[data['date']!=last_date].copy()
valid_data = data[data['date']==last_date].copy()
train_data = train_data.drop(columns=['date'])
valid_data = valid_data.drop(columns=['date'])
# print(f"训练集形状{train_data.shape}，平均点击率{train_data['clk'].mean()}")
# print(f"验证集形状{valid_data.shape}，平均点击率{valid_data['clk'].mean()}")

# 进一步切分类别、等级和标签
choice = 1
if choice == 1:
    cate_cols = ['cms_segid','hour','weekday','pid','cate_id','cms_group_id','final_gender_code','occupation','brand']
    num_cols = ['price','age_level','shopping_level','pvalue_level','new_user_class_level']
elif choice == 2:
    cate_cols = ['cms_segid','hour','weekday','pid','cate_id','cms_group_id','final_gender_code','occupation']
    num_cols = ['age_level','shopping_level','pvalue_level','new_user_class_level']
else:
    cate_cols = ['hour','weekday','pid','cate_id','cms_group_id','final_gender_code','occupation','brand']
    num_cols = ['price','age_level','shopping_level']

feat_cols = cate_cols + num_cols # type: ignore
x_train = train_data[feat_cols]
y_train = train_data['clk']
x_valid = valid_data[feat_cols]
y_valid = valid_data['clk']
# print(f"\n训练集特征形状{x_train.shape},标签形状{y_train.shape}")
# print(f"验证集特征形状{x_valid.shape},标签形状{y_valid.shape}")
print("处理完成!")
print(f"当前训练集形状{x_train.shape},验证集形状{x_valid.shape}\n")


from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score,log_loss

print("开始训练LGBM模型...")
n_estimators = 32
model = LGBMClassifier(objective='binary', random_state=13, n_estimators=n_estimators,learning_rate=0.1, num_leaves=3)
model.fit(x_train,y_train,categorical_feature=cate_cols)
print(f"模型训练完成，树个数为{n_estimators}")

y_predict = model.predict_proba(x_valid)
score = roc_auc_score(y_true=y_valid,y_score=y_predict[:,1]) # type: ignore
loss = log_loss(y_true=y_valid,y_pred=y_predict) # type: ignore
print(f"验证集AUC:{score}\n验证集logloss:{loss}")
