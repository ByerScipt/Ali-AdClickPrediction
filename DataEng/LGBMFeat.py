
'''
以sample为主表，将广告和用户的特征接入，保留并构造有用的特征，同时处理缺失值
'''

import pandas as pd
from pathlib import Path

file_path = "../data/raw_data/"
ad_path = "ad_feature.csv"
sample_path = "raw_sample.csv"
user_path = "user_profile.csv"

# 读数据
sample = pd.read_csv(file_path+sample_path,nrows=100000)
# 只保留使用的特征列
sample = sample[['time_stamp','clk','pid','user','adgroup_id']]

ad = pd.read_csv(file_path+ad_path)
usr = pd.read_csv(file_path+user_path)
usr.columns = usr.columns.str.strip()  # 清除new_user_class_level名称后的一个空格
print(f"初始sample的形状{sample.shape}")

# 合并广告和用户特征
sample = sample.merge(ad,on='adgroup_id',how='left')
sample = sample.merge(usr,left_on='user',right_on='userid',how='left')
keep_cols = ['time_stamp','clk','pid','user','adgroup_id','cate_id','cms_group_id','cms_segid','final_gender_code','occupation','age_level','shopping_level','brand','price','pvalue_level','new_user_class_level']
sample = sample[keep_cols]
print(f"\n合并广告和用户特征后的形状{sample.shape}")
print(f"合并广告和用户特征后的字段{sample.columns}")
print(f"数据类型：\n{sample.dtypes}")

# 处理缺失
print(f"\n数据缺失率：\n{sample.isna().mean().sort_values(ascending=False)}")
cols = ['cms_segid','pvalue_level','new_user_class_level','brand','occupation','shopping_level','age_level','final_gender_code','cms_group_id']
sample[cols] = sample[cols].fillna(-1)

# 提取时间信息
time_stamp = pd.to_datetime(sample['time_stamp'],unit='s')
sample['hour'] = time_stamp.dt.hour
sample['weekday'] = time_stamp.dt.weekday
sample['date'] = time_stamp.dt.date

# 数据增强
sample = sample.sort_values('time_stamp')
# 提取商品历史曝光与点击数
sample['ad_hist_imp'] = sample.groupby('adgroup_id').cumcount()
sample['tmp_clk_cumsum'] = sample.groupby('adgroup_id')['clk'].cumsum()
sample['ad_hist_clk'] = sample.groupby('adgroup_id')['tmp_clk_cumsum'].shift(1)
sample['ad_hist_clk'] = sample['ad_hist_clk'].fillna(0)
sample['ad_hist_ctr'] = (sample['ad_hist_clk']+0.5) / (sample['ad_hist_imp']+10) # 根据统计，每日点击率在0.05左右波动，故选择此数值进行平滑
# 提取用户历史曝光与点击数
sample['user_hist_imp'] = sample.groupby('user').cumcount()
sample['tmp_clk_cumsum'] = sample.groupby('user')['clk'].cumsum()
sample['user_hist_clk'] = sample.groupby('user')['tmp_clk_cumsum'].shift(1)
sample['user_hist_clk'] = sample['user_hist_clk'].fillna(0)
sample['user_hist_ctr'] = (sample['user_hist_clk']+0.25) / (sample['user_hist_imp']+5) 
# 用户×商品种类交叉
sample['user_cate_hist_imp'] = sample.groupby(['user','cate_id']).cumcount()
sample['tmp_clk_cumsum'] = sample.groupby(['user','cate_id'])['clk'].cumsum()
sample['user_cate_hist_clk'] = sample.groupby(['user','cate_id'])['tmp_clk_cumsum'].shift(1)
sample['user_cate_hist_clk'] = sample['user_cate_hist_clk'].fillna(0)
sample['user_cate_hist_ctr'] = (sample['user_cate_hist_clk']+0.25) / (sample['user_cate_hist_imp']+5)
sample = sample.drop(columns=['tmp_clk_cumsum','time_stamp'])

# 将缺失类别建模成未知类别
sample[['brand','occupation','cms_group_id','final_gender_code','cms_segid']] += 1

# 将pid映射成整数类别
sample['pid'] = sample['pid'].astype('category').cat.codes

# 保存
Path('../data/processed_data').mkdir(parents=True,exist_ok=True)
sample.to_parquet('../data/sample/LGBMSample.parquet',index=False)
