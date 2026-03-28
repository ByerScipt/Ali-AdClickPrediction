
'''
以sample为主表，将广告和用户的特征接入，构成唯一的宽表，同时处理缺失值
'''

import pandas as pd
from pathlib import Path

file_path = "../data/raw_data/"
ad_path = "ad_feature.csv"
sample_path = "raw_sample.csv"
user_path = "user_profile.csv"

# 读数据
sample = pd.read_csv(file_path+sample_path,nrows=100000)
ad = pd.read_csv(file_path+ad_path)
usr = pd.read_csv(file_path+user_path)
usr.columns = usr.columns.str.strip()  # 清除new_user_class_level名称后的一个空格
print(f"初始sample的形状{sample.shape}")

# 合并广告和用户特征
sample = sample.merge(ad,on='adgroup_id',how='left')
sample = sample.merge(usr,left_on='user',right_on='userid',how='left')
sample = sample.drop(columns=['userid','nonclk'])
print(f"\n合并广告和用户特征后的形状{sample.shape}")
print(f"合并广告和用户特征后的字段{sample.columns}")
print(f"数据类型：\n{sample.dtypes}")

# 处理缺失
print(f"\n数据缺失率：\n{sample.isna().mean().sort_values(ascending=False)}")
cols = ['pvalue_level','new_user_class_level','brand','cms_segid','occupation','shopping_level','age_level','final_gender_code','cms_group_id']
sample[cols] = sample[cols].fillna(-1)

# 保存
Path('../data/processed_data').mkdir(parents=True,exist_ok=True)
sample.to_csv('../data/processed_data/LRSample.csv')
