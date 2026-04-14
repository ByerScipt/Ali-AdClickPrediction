import pandas as pd
from pathlib import Path
import pyarrow.dataset as ds
import duckdb
from collections import defaultdict
from bisect import bisect_left

file_path = "../data/raw_data/"
processed_path = "../data/processed_data/"
save_path = "../data/sample/"
ad_path = "ad_feature.csv"
sample_path = "raw_sample.csv"
user_path = "user_profile.csv"
beh_path = "../data/processed_data/behavior_log.parquet"

# 用户行为只截取最近100条，减少运算量
# 根据数据分析，每个曝光样本对应用户在曝光时间之前的历史序列长度中位数为482
max_beh_len = 100  

sample = pd.read_csv(file_path+sample_path,nrows=100000)
sample = sample[['time_stamp','clk','pid','user','adgroup_id']]  # 依旧去掉没用的nonclk特征

ad = pd.read_csv(file_path+ad_path)
usr = pd.read_csv(file_path+user_path)
usr.columns = usr.columns.str.strip()
print("-------读取并处理静态特征-------")
print(f"初始sample的形状{sample.shape}")

# 合并广告和用户特征
sample = sample.merge(ad,on='adgroup_id',how='left')
sample = sample.merge(usr,left_on='user',right_on='userid',how='left')
keep_cols = ['time_stamp','clk','pid','user','adgroup_id','cate_id','cms_group_id','cms_segid','final_gender_code','occupation','age_level','shopping_level','brand','price']
sample = sample[keep_cols]

# 处理缺失
# print(f"数据缺失率：\n{sample.isna().mean().sort_values(ascending=False)}")
cols = ['cms_segid','brand','occupation','shopping_level','age_level','final_gender_code','cms_group_id']
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
# 提取用户历史曝光与点击数
sample['user_hist_imp'] = sample.groupby('user').cumcount()
sample['tmp_clk_cumsum'] = sample.groupby('user')['clk'].cumsum()
sample['user_hist_clk'] = sample.groupby('user')['tmp_clk_cumsum'].shift(1)
sample['user_hist_clk'] = sample['user_hist_clk'].fillna(0)
sample['user_hist_ctr'] = (sample['user_hist_clk']+0.25) / (sample['user_hist_imp']+5) # 根据统计，每日点击率在0.05左右波动，故选择此数值进行平滑
# 用户×商品种类交叉
sample['user_cate_hist_imp'] = sample.groupby(['user','cate_id']).cumcount()
sample['tmp_clk_cumsum'] = sample.groupby(['user','cate_id'])['clk'].cumsum()
sample['user_cate_hist_clk'] = sample.groupby(['user','cate_id'])['tmp_clk_cumsum'].shift(1)
sample['user_cate_hist_clk'] = sample['user_cate_hist_clk'].fillna(0)
sample['user_cate_hist_ctr'] = (sample['user_cate_hist_clk']+0.25) / (sample['user_cate_hist_imp']+5)
sample = sample.drop(columns=['tmp_clk_cumsum'])
# 显式交叉
sample['gender_cate_cross'] = (sample['final_gender_code'].astype(str)+'_'+sample['cate_id'].astype(str))
print("静态特征处理完成！")
print(f"处理后的sample形状{sample.shape}")
print(f"处理后的sample字段{sample.columns}\n")


# 开始构造用户行为序列
print("-------读取并处理行为日志-------")

# 使用DuckDB做过滤，只保留sample中出现的用户的行为序列
sample_user = set(sample['user'].unique())
beh_sub_path = Path(processed_path) / f"behavior_log_din_{len(sample)}_{len(sample_user)}.parquet"

if not beh_sub_path.exists():
    print("使用DuckDB预过滤行为日志...")
    con = duckdb.connect()
    sample_user_df = pd.DataFrame({'user': sorted(sample_user)})
    con.register('sample_user_df', sample_user_df)
    con.execute(f"""
        COPY (
            SELECT b.user, b.time_stamp, b.btag, b.cate, b.brand
            FROM read_parquet('{Path(beh_path).resolve()}') AS b
            INNER JOIN sample_user_df AS u
            ON b.user = u.user
            ORDER BY b.user, b.time_stamp
        ) TO '{beh_sub_path.resolve()}' (FORMAT PARQUET)
    """)
    con.close()
    print(f"过滤后的行为日志已保存到: {beh_sub_path}\n")
else:
    print(f"直接复用过滤后的行为日志: {beh_sub_path}\n")

# 
print("将行为序列以用户为单位分组...")
dataset = ds.dataset(beh_sub_path, format='parquet')
hist_beh = defaultdict(list)
hist_time = defaultdict(list)
scanner = dataset.scanner(
    columns=['user', 'time_stamp', 'btag', 'cate', 'brand'],
    batch_size=100000
)
for batch_id, batch in enumerate(scanner.to_batches(), start=1):
    chunk = batch.to_pandas()
    if len(chunk) == 0: continue
    chunk = chunk.sort_values(['user', 'time_stamp'])
    for user, group in chunk.groupby('user', sort=False):
        time_list = group['time_stamp'].tolist()
        hist_time[user].extend(time_list)
        hist_beh[user].extend(
            zip(
                time_list,
                group['cate'].tolist(),
                group['brand'].tolist(),
                group['btag'].tolist()
            )
        )
    if batch_id % 200 == 0:
        print(f'已处理batch:{batch_id}, 当前命中用户数:{len(hist_beh)}')
print(f"最终命中用户数: {len(hist_beh)}\n")

# 为每个曝光样本附加一个用户历史序列
print("为每个曝光样本构造行为序列...")
seq_rows = []
for _,row in sample.iterrows():
    user = row['user']
    expo_time = row['time_stamp']
    user_beh = hist_beh.get(user,[]) # 如果在hist_beh找不到，就返回空列表
    beh_times = hist_time.get(user,[])
    cut_idx = bisect_left(beh_times,expo_time) # 二分查找第一个大于等于曝光时间的位置
    recent_beh = user_beh[max(0,cut_idx-max_beh_len):cut_idx] # 最多只保留max_beh_len长度的历史行为
    # 合并行为序列信息
    hist_cate_seq = [e[1] for e in recent_beh]
    hist_brand_seq = [e[2] for e in recent_beh]
    hist_btag_seq = [e[3] for e in recent_beh]
    seq_rows.append({
        'hist_cate_seq': hist_cate_seq,
        'hist_brand_seq': hist_brand_seq,
        'hist_btag_seq': hist_btag_seq,
        'seq_len': len(recent_beh)
    })
print("行为序列构造完成！")  

# 拼接回sample
sample = sample.reset_index(drop=True) # 恢复sample原来的排序
seq_df = pd.DataFrame(seq_rows)
sample = pd.concat([sample,seq_df],axis=1)
print(f"样本行为序列长度分布:\n{sample['seq_len'].describe()}")

out_path = Path(save_path) / 'DINSample.parquet'
sample.to_parquet(out_path, index=False)
print(f'DIN样本已保存到: {out_path}')