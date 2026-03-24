import pandas as pd

file_path = "/home/byer/AliCCP/data/raw_data/"
ad_path = "ad_feature.csv"
sample_path = "raw_sample.csv"
user_path = "user_profile.csv"

# 阅读raw_sample的结构
print("---------------raw_sample-------------------")
sample = pd.read_csv(file_path+sample_path,nrows=100000)
print(sample.shape)
print(sample.columns)
print(sample.dtypes)
print(sample.head(n=5))

print(sample["clk"].value_counts())
print(sample["clk"].value_counts(normalize=True))

print(f"用户ID去重数：{sample['user'].nunique()}")
print(f"广告ID去重数：{sample['adgroup_id'].nunique()}")
print(f"资源位PID去重数：{sample['pid'].nunique()}")

print(sample.groupby('pid')['clk'].agg(['count','mean']))
time_stamp = pd.to_datetime(sample["time_stamp"],unit='s')
sample['date'] = time_stamp.dt.date
print("\n时间统计：")
print(f"最小值{time_stamp.min()}")
print(f"最大值{time_stamp.max()}")
print("按日统计CTR：")
print(sample.groupby('date')['clk'].agg(['count','mean']))


# 阅读ad_feature
print("\n\n---------------ad_feature-------------------")
ad = pd.read_csv(file_path+ad_path)
print(ad.columns)
print(ad.dtypes)
print(ad.head(n=5))
print(f"品牌缺失率：{ad['brand'].isna().mean()}")
print(f"价格缺失率：{ad['price'].isna().mean()}")
print(f"广告覆盖率：{sample['adgroup_id'].isin(ad['adgroup_id']).mean()}")
print(f"广告覆盖率：{sample['adgroup_id'].drop_duplicates().isin(ad['adgroup_id']).mean()}")
print(f"广告数：{len(ad['adgroup_id'])}；去重数：{ad['adgroup_id'].nunique()}")

# 阅读user_profile
print("\n\n---------------user_profile-------------------")
usr = pd.read_csv(file_path+user_path)
usr.columns = usr.columns.str.strip()
print(usr.columns)
print(usr.dtypes)
print(usr.head(n=5))
print(f"消费档次缺失率：{usr['pvalue_level'].isna().mean()}")
print(f"微群缺失率：{usr['cms_segid'].isna().mean()}")
print(f"微群组缺失率：{usr['cms_group_id'].isna().mean()}")
print(f"消费层级缺失率：{usr['shopping_level'].isna().mean()}")
print(f"性别信息缺失率：{usr['final_gender_code'].isna().mean()}")
print(f"是否是大学生缺失率：{usr['occupation'].isna().mean()}")
print(f"用户城市层级缺失率：{usr['new_user_class_level'].isna().mean()}")
print(f"用户覆盖率：{sample['user'].isin(usr['userid']).mean()}")
print(f"用户覆盖率：{sample['user'].drop_duplicates().isin(usr['userid']).mean()}")
print(f"用户数：{len(usr['userid'])}；去重数：{usr['userid'].nunique()}")