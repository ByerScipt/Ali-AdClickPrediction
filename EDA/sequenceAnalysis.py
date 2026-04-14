import pandas as pd
import pyarrow.parquet as pq
from collections import defaultdict
from bisect import bisect_left

file_path = "../data/raw_data/"
ad_path = "ad_feature.csv"
sample_path = "raw_sample.csv"
behavior_path = "../data/processed_data/behavior_log.parquet"

sample = pd.read_csv(file_path+sample_path,nrows=100000)
sample = sample[['user','time_stamp','adgroup_id']]
print(f"曝光样本用户去重数{sample['user'].nunique()}")

ad = pd.read_csv(file_path+ad_path)
ad = ad[['cate_id','brand','adgroup_id']]

sample = sample.merge(ad,on='adgroup_id',how='left')
print(f"广告类目缺失率{sample['cate_id'].isna().mean()}")
print(f"广告品牌缺失率{sample['brand'].isna().mean()}")
print(f"广告覆盖率{sample['adgroup_id'].isin(ad['adgroup_id']).mean()}")

print("\n----------用户行为数据集分析---------")
users = set(sample['user'].unique())
hit_rows = 0
hit_user = set()
brand_missing_cnt = 0
btag_cnt = defaultdict(int)
beh_user_cnt = defaultdict(int)
pf = pq.ParquetFile(behavior_path)

print("\n----曝光前历史长度分析----")
sample_small = sample[['user','time_stamp']].sample(n=2000,random_state=42).copy()
sample_small = sample_small.sort_values(['user','time_stamp'])
sample_small_users = set(sample_small['user'].unique())
hist_times = defaultdict(list)

for batch in pf.iter_batches(batch_size=100000,columns=['user','time_stamp','btag','cate','brand']):
    chunk = batch.to_pandas()
    chunk = chunk[chunk['user'].isin(users)]
    chunk_small = chunk[chunk['user'].isin(sample_small_users)]
    if len(chunk_small) > 0:
        chunk_small_user_times = chunk_small.groupby('user')['time_stamp'].apply(list)
        for key,value in chunk_small_user_times.items():
            hist_times[key].extend(value)
    if len(chunk) == 0: continue
    chunk_user_cnt = chunk['user'].value_counts()
    for key,value in chunk_user_cnt.items():
        beh_user_cnt[key] += value
    hit_rows += len(chunk)
    hit_user.update(chunk['user'].unique())
    brand_missing_cnt += chunk['brand'].isna().sum()
    chunk_btag = chunk['btag'].value_counts(dropna=False)
    for key,value in chunk_btag.items():
        btag_cnt[key] += value
          
hist_len_list = []
matched_expo = 0
for _,row in sample_small.iterrows():
    user = row['user']
    expo_time = row['time_stamp']
    if user not in hist_times: continue
    matched_expo += 1
    hist_len = bisect_left(hist_times[user],expo_time)
    hist_len_list.append(hist_len)
hist_len_list = pd.Series(hist_len_list)
print(f"sample_small曝光数{len(sample_small)}")
print(f"成功匹配历史的曝光数{matched_expo}")
print("曝光前历史长度分位数：")
print(hist_len_list.quantile([0.5,0.75,0.9,0.95,0.99]))
print(f"曝光前历史长度大于20的占比{(hist_len_list > 20).mean()}")
print(f"曝光前历史长度大于50的占比{(hist_len_list > 50).mean()}")
print(f"曝光前历史长度大于100的占比{(hist_len_list > 100).mean()}")
print(f"曝光前历史长度大于200的占比{(hist_len_list > 200).mean()}")

print("\n----用户行为条数分析----")
beh_user_cnt = pd.Series(beh_user_cnt)  
print("命中用户总行为条数分位数：")
print(beh_user_cnt.quantile([0.5,0.75,0.9,0.95,0.99]))
print(f"用户平均行为条数{beh_user_cnt.mean()}")
print(f"用户最大行为条数{beh_user_cnt.max()}")
print(f"总行为条数大于20的用户占比{(beh_user_cnt > 20).mean()}")
print(f"总行为条数大于50的用户占比{(beh_user_cnt > 50).mean()}")
print(f"总行为条数大于100的用户占比{(beh_user_cnt > 100).mean()}")
print(f"命中行为总行数{hit_rows}")
print(f"命中用户数{len(hit_user)}")
print(f"曝光用户历史覆盖率{len(hit_user)/len(users)}")
print(f"行为序列品牌缺失率{brand_missing_cnt/hit_rows}")
print("btag分布：")
print(btag_cnt)

