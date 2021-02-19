#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2020.06.08

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
from functools import partial


from utils.tools import parallel_apply
from utils.feature_extraction import add_features_in_group
def last_k_call_interval(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['start_datetime'], inplace=True)
    gr_['call_interval'] = gr_['start_datetime'].diff()
    gr_['call_interval'] = gr_['call_interval'].dt.total_seconds()
    gr_['call_interval'] = gr_['call_interval'].fillna(0)
    features = {}
    for period in periods:
        if period > 10e10:
            period_name = 'voc_all'
            gr_period = gr_.copy()
        else:
            period_name = 'voc_last_{}_'.format(period)
            gr_period = gr_.iloc[:period]

        features = add_features_in_group(features, gr_period, 'call_interval',
                                         ['mean','max', 'min', 'std'],
                                         # ['diff'],
                                         period_name)
    return features
def last_k_caller_interval(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['start_datetime'], inplace=True)
    gr_=gr_[gr_['calltype_id']==1]
    gr_['caller_interval'] = gr_['start_datetime'].diff()
    gr_['caller_interval'] = gr_['caller_interval'].dt.total_seconds()
    gr_['caller_interval'] = gr_['caller_interval'].fillna(0)
    features = {}
    for period in periods:
        if period > 10e10:
            period_name = 'voc_all'
            gr_period = gr_.copy()
        else:
            period_name = 'voc_last_{}_'.format(period)
            gr_period = gr_.iloc[:period]

        features = add_features_in_group(features, gr_period, 'caller_interval',
                                         ['mean','max', 'min', 'std'],
                                         # ['diff'],
                                         period_name)
    return features

def user_voc_fea(df_voc):
    # # 异常值
    # df_voc.loc[df_voc['call_dur']>3600 ,'call_dur']= np.nan
    #
    # df_voc = df_voc.drop(['opposite_no_m','imei_m'],axis = 1).sort_values(by=['phone_no_m', 'start_datetime'],ascending = False)
    df_voc['start_datetime'] = pd.to_datetime(df_voc['start_datetime'])
    df_voc['voc_day'] = df_voc['start_datetime'].dt.day
    df_voc['voc_hour'] = df_voc['start_datetime'].dt.hour
    df_voc['voc_day_mean'] = df_voc.groupby(['phone_no_m'])['voc_day'].transform('mean')
    df_voc['voc_hour_mean'] = df_voc.groupby(['phone_no_m'])['voc_hour'].transform('mean')

    # 每月电话量统计
    df_voc['voc_month_count'] = df_voc.groupby(['phone_no_m'])['phone_no_m'].transform('count')
    # 每天电话量统计
    df_voc['voc_day_count'] = df_voc.groupby(['phone_no_m', 'voc_day'])['phone_no_m'].transform('count')
    df_voc['voc_day_count_max'] = df_voc.groupby('phone_no_m')['voc_day_count'].transform('max')
    df_voc.loc[(df_voc['voc_day_count_max'] > 150) &(df_voc['voc_day_count_max'] < 250), 'voc_day_count_max'] = (150 + 250) / 2
    df_voc['voc_day_count_min'] = df_voc.groupby('phone_no_m')['voc_day_count'].transform('min')
    df_voc.loc[(df_voc['voc_day_count_min'] > 180) &(df_voc['voc_day_count_min'] < 200), 'voc_day_count_min'] = (180 + 200) / 2
    df_voc['voc_day_count_mean'] = df_voc.groupby('phone_no_m')['voc_day_count'].transform('mean')
    df_voc['voc_day_count_std'] = df_voc.groupby('phone_no_m')['voc_day_count'].transform('std')
    del df_voc['voc_day_count']
    # 每小时电话量统计
    df_voc['voc_hour_count'] = df_voc.groupby(['phone_no_m', 'voc_hour'])['phone_no_m'].transform('count')
    df_voc['voc_hour_count_max'] = df_voc.groupby('phone_no_m')['voc_hour_count'].transform('max')
    df_voc.loc[(df_voc['voc_hour_count_max'] > 400) &(df_voc['voc_hour_count_max'] < 500), 'voc_hour_count_max'] = (400 + 500) / 2
    df_voc['voc_hour_count_min'] = df_voc.groupby('phone_no_m')['voc_hour_count'].transform('min')
    df_voc.loc[(df_voc['voc_hour_count_min'] > 35) &(df_voc['voc_hour_count_min'] < 50), 'voc_hour_count_min'] = (35 + 50) / 2
    df_voc['voc_hour_count_mean'] = df_voc.groupby('phone_no_m')['voc_hour_count'].transform('mean')
    df_voc.loc[(df_voc['voc_hour_count_mean'] > 300) &(df_voc['voc_hour_count_mean'] < 400), 'voc_hour_count_mean'] = (300 + 400) / 2
    df_voc['voc_hour_count_std'] = df_voc.groupby('phone_no_m')['voc_hour_count'].transform('std')
    del df_voc['voc_hour_count']
    # 每月电话类型统计
    df_voc['voc_month_type_count'] = df_voc.groupby(['phone_no_m'])['calltype_id'].transform('count')
    df_voc.loc[df_voc['voc_month_type_count'] > 2700, 'voc_month_type_count'] = 2700
    # 每天电话类型统计
    df_voc['voc_day_type_count'] = df_voc.groupby(['phone_no_m', 'voc_day'])['calltype_id'].transform('count')
    df_voc['voc_day_type_count_max'] = df_voc.groupby('phone_no_m')['voc_day_type_count'].transform('max')
    df_voc.loc[(df_voc['voc_day_type_count_max'] > 250) &(df_voc['voc_day_type_count_max'] < 350), 'voc_day_type_count_max'] = (250 + 350) / 2
    df_voc['voc_day_type_count_min'] = df_voc.groupby('phone_no_m')['voc_day_type_count'].transform('min')
    df_voc.loc[(df_voc['voc_day_type_count_min'] > 150) &(df_voc['voc_day_type_count_min'] < 250), 'voc_day_type_count_min'] = (150 + 250) / 2
    df_voc['voc_day_type_count_mean'] = df_voc.groupby('phone_no_m')['voc_day_type_count'].transform('mean')
    df_voc.loc[(df_voc['voc_day_type_count_mean'] > 200) &(df_voc['voc_day_type_count_mean'] < 250), 'voc_day_type_count_mean'] = (200 + 250) / 2
    df_voc['voc_day_type_count_std'] = df_voc.groupby('phone_no_m')['voc_day_type_count'].transform('std')
    del df_voc['voc_day_type_count']

    # 每月电话时长分段统计
    df_voc['call_dur_cut'] = 0
    df_voc.loc[df_voc['call_dur']<10,'call_dur_cut'] = 0
    df_voc.loc[(df_voc['call_dur'] >= 10) & (df_voc['call_dur'] < 50),'call_dur_cut'] = 1
    df_voc.loc[(df_voc['call_dur'] >= 50) & (df_voc['call_dur'] < 150),'call_dur_cut'] = 2
    df_voc.loc[(df_voc['call_dur'] >= 150) & (df_voc['call_dur'] < 300),'call_dur_cut'] = 3
    df_voc.loc[df_voc['call_dur'] >= 300,'call_dur_cut'] = 4
    df_voc['voc_month_call_dur_count'] = df_voc.groupby(['phone_no_m', 'call_dur_cut'])['call_dur_cut'].transform(
        'count')
    df_voc['voc_month_call_dur_count_max'] = df_voc.groupby(['phone_no_m'])['voc_month_call_dur_count'].transform('max')
    df_voc.loc[(df_voc['voc_month_call_dur_count_max'] > 1500), 'voc_month_call_dur_count_max'] = 1500
    df_voc['voc_month_call_dur_count_min'] = df_voc.groupby('phone_no_m')['voc_month_call_dur_count'].transform('min')
    df_voc.loc[(df_voc['voc_month_call_dur_count_min'] > 100), 'voc_month_call_dur_count_min'] = 10
    df_voc['voc_month_call_dur_count_mean'] = df_voc.groupby('phone_no_m')['voc_month_call_dur_count'].transform('mean')
    df_voc.loc[(df_voc['voc_month_call_dur_count_mean'] > 1200), 'voc_month_call_dur_count_mean'] = 1200
    df_voc['voc_month_call_dur_count_std'] = df_voc.groupby('phone_no_m')['voc_month_call_dur_count'].transform('std')
    df_voc.loc[(df_voc['voc_month_call_dur_count_std'] > 590), 'voc_month_call_dur_count_std'] = 590

    # 每天电话时长分段统计
    df_voc['voc_day_call_dur_count'] = df_voc.groupby(['phone_no_m', 'voc_day', 'call_dur_cut'])[
        'call_dur_cut'].transform('count')
    df_voc['voc_day_call_dur_count_max'] = df_voc.groupby(['phone_no_m'])['voc_day_call_dur_count'].transform('max')
    df_voc.loc[(df_voc['voc_day_call_dur_count_max'] > 190), 'voc_day_call_dur_count_max'] = 190
    df_voc['voc_day_call_dur_count_min'] = df_voc.groupby('phone_no_m')['voc_day_call_dur_count'].transform('min')
    df_voc['voc_day_call_dur_count_mean'] = df_voc.groupby('phone_no_m')['voc_day_call_dur_count'].transform('mean')
    df_voc['voc_day_call_dur_count_std'] = df_voc.groupby('phone_no_m')['voc_day_call_dur_count'].transform('std')
    df_voc.loc[(df_voc['voc_day_call_dur_count_std'] > 70), 'voc_day_call_dur_count_std'] = 70


    del df_voc['call_dur_cut']
    del df_voc['voc_month_call_dur_count']
    del df_voc['voc_day_call_dur_count']
    # # 每月加密号码电话统计
    df_voc['voc_month_imei_m_nunique'] = df_voc.groupby(['phone_no_m'])['imei_m'].transform('nunique')
    df_voc.loc[(df_voc['voc_month_imei_m_nunique'] > 30), 'voc_month_imei_m_nunique'] = 30

    # 每天加密电话种类统计
    df_voc['voc_day_imei_m_nunique'] = df_voc.groupby(['phone_no_m', 'voc_day'])['imei_m'].transform('nunique')
    df_voc['voc_day_imei_m_count_max'] = df_voc.groupby(['phone_no_m'])['voc_day_imei_m_nunique'].transform('max')
    df_voc.loc[(df_voc['voc_day_imei_m_count_max'] > 8), 'voc_day_imei_m_count_max'] = 8
    df_voc['voc_day_imei_m_count_min'] = df_voc.groupby('phone_no_m')['voc_day_imei_m_nunique'].transform('min')
    df_voc.loc[(df_voc['voc_day_imei_m_count_min'] > 2), 'voc_day_imei_m_count_min'] = 2

    df_voc['voc_day_imei_m_count_mean'] = df_voc.groupby('phone_no_m')['voc_day_imei_m_nunique'].transform('mean')
    df_voc.loc[(df_voc['voc_day_imei_m_count_mean'] > 5), 'voc_day_imei_m_count_mean'] = 5
    df_voc['voc_day_imei_m_count_std'] = df_voc.groupby('phone_no_m')['voc_day_imei_m_nunique'].transform('std')
    df_voc.loc[(df_voc['voc_day_imei_m_count_std'] > 1.5), 'voc_day_imei_m_count_std'] = 1.5

    del df_voc['voc_day_imei_m_nunique']
    # # 每月对端加密电话数量统计
    df_voc['voc_month_opposite_no_m_nunique'] = df_voc.groupby(['phone_no_m'])['opposite_no_m'].transform('nunique')
    df_voc.loc[(df_voc['voc_month_opposite_no_m_nunique'] > 1150) &(df_voc['voc_month_opposite_no_m_nunique'] < 1800), 'voc_month_opposite_no_m_nunique'] = (1250 + 1500) / 2

    # # 每天对端加密电话数量统计
    df_voc['voc_day_opposite_no_m_nunique'] = df_voc.groupby(['phone_no_m', 'voc_day'])['opposite_no_m'].transform(
        'nunique')
    df_voc['voc_day_opposite_no_m_count_max'] = df_voc.groupby(['phone_no_m'])[
        'voc_day_opposite_no_m_nunique'].transform('max')
    df_voc.loc[(df_voc['voc_day_opposite_no_m_count_max'] > 150) &(df_voc['voc_day_opposite_no_m_count_max'] < 200), 'voc_day_opposite_no_m_count_max'] = (150 + 200) / 2
    df_voc['voc_day_opposite_no_m_count_min'] = df_voc.groupby('phone_no_m')['voc_day_opposite_no_m_nunique'].transform(
        'min')
    df_voc.loc[(df_voc['voc_day_opposite_no_m_count_min'] > 150) &(df_voc['voc_day_opposite_no_m_count_min'] < 200), 'voc_day_opposite_no_m_count_min'] = (150 + 200) / 2
    df_voc['voc_day_opposite_no_m_count_mean'] = df_voc.groupby('phone_no_m')[
        'voc_day_opposite_no_m_nunique'].transform('mean')
    df_voc.loc[(df_voc['voc_day_opposite_no_m_count_mean'] > 150) &(df_voc['voc_day_opposite_no_m_count_mean'] < 200), 'voc_day_opposite_no_m_count_mean'] = (150 + 200) / 2

    df_voc['voc_day_opposite_no_m_count_std'] = df_voc.groupby('phone_no_m')['voc_day_opposite_no_m_nunique'].transform(
        'std')
    df_voc.loc[(df_voc['voc_day_opposite_no_m_count_std'] > 60) &(df_voc['voc_day_opposite_no_m_count_std'] < 70), 'voc_day_opposite_no_m_count_std'] = (60 + 70) / 2

    del df_voc['voc_day_opposite_no_m_nunique']

    # df_voc['ga'] = 0
    # df_voc.loc[df_voc['city_name']=='广安','ga'] = 1
    # # df_voc['tfxq'] = 0
    # # df_voc.loc[df_voc['city_name'] == '天府新区', 'tfxq'] = 1
    # # df_voc['dy'] = 0
    # # df_voc.loc[df_voc['city_name'] == '德阳', 'dy'] = 1
    # # df_voc['cd'] = 0
    # # df_voc.loc[df_voc['city_name'] == '成都', 'cd'] = 1
    # # df_voc['bz'] = 0
    # # df_voc.loc[df_voc['city_name'] == '巴中', 'bz'] = 1
    # # df_voc['ls'] = 0
    # # df_voc.loc[df_voc['city_name'] == '乐山', 'ls'] = 1
    #
    # df_voc['voc_city_name_ga'] = df_voc.groupby(['phone_no_m'])['ga'].transform('sum')
    #
    # del df_voc['ga']



    # 地理、区县编码后做统计特征
    for f in tqdm(['city_name', 'county_name']):
        lbl = LabelEncoder()
        df_voc[f] = df_voc[f].fillna('NA')
        df_voc[f] = lbl.fit_transform(df_voc[f].astype(str))
    # 每月的的地理变化统计
    df_voc['voc_month_call_city_name_nunique'] = df_voc.groupby(['phone_no_m'])['city_name'].transform('nunique')
    df_voc['voc_month_call_city_name_mean'] = df_voc.groupby(['phone_no_m'])['city_name'].transform('mean')
    df_voc['voc_month_call_city_name_size'] = df_voc.groupby(['phone_no_m'])['city_name'].transform('size')
    df_voc.loc[(df_voc['voc_month_call_city_name_size'] > 2500), 'voc_month_call_city_name_size'] = 2500

    df_voc['voc_month_call_city_name_std'] = df_voc.groupby(['phone_no_m'])['city_name'].transform('std')
    # df_voc.loc[(df_voc['voc_month_call_city_name_std'] > 60) &(df_voc['voc_month_call_city_name_std'] < 70), 'voc_day_opposite_no_m_count_std'] = (60 + 70) / 2

    df_voc['voc_month_call_county_name_nunique'] = df_voc.groupby(['phone_no_m'])['county_name'].transform('nunique')
    df_voc.loc[(df_voc['voc_month_call_county_name_nunique'] > 30) &(df_voc['voc_month_call_county_name_nunique'] < 50),
               'voc_month_call_county_name_nunique'] = (30 + 50) / 2

    df_voc['voc_month_call_county_name_mean'] = df_voc.groupby(['phone_no_m'])['county_name'].transform('mean')
    df_voc['voc_month_call_county_name_size'] = df_voc.groupby(['phone_no_m'])['county_name'].transform('size')
    df_voc.loc[(df_voc['voc_month_call_county_name_size'] > 2500), 'voc_month_call_county_name_size'] = 2500
    df_voc['voc_month_call_county_name_std'] = df_voc.groupby(['phone_no_m'])['county_name'].transform('std')
    # 每天的地理变化统计
    # df_voc['voc_day_call_city_name_nunique'] = df_voc.groupby(['phone_no_m','voc_day'])['city_name'].transform('nunique')
    # df_voc['voc_day_call_city_name_mean'] = df_voc.groupby(['phone_no_m','voc_day'])['city_name'].transform('mean')
    # df_voc['voc_day_call_city_name_size'] = df_voc.groupby(['phone_no_m','voc_day'])['city_name'].transform('size')
    # df_voc['voc_day_call_city_name_var'] = df_voc.groupby(['phone_no_m','voc_day'])['city_name'].transform('var')
    # df_voc['voc_day_call_county_name_nunique'] = df_voc.groupby(['phone_no_m','voc_day'])['county_name'].transform('nunique')
    # df_voc['voc_day_call_county_name_mean'] = df_voc.groupby(['phone_no_m','voc_day'])['county_name'].transform('mean')
    # df_voc['voc_day_call_county_name_size'] = df_voc.groupby(['phone_no_m','voc_day'])['county_name'].transform('size')
    # df_voc['voc_day_call_county_name_var'] = df_voc.groupby(['phone_no_m','voc_day'])['county_name'].transform('var')
    df_voc['voc_day_call_city_name_nunique'] = df_voc.groupby(['phone_no_m', 'voc_day'])['city_name'].transform('nunique')
    df_voc['voc_day_call_city_name_mean'] = df_voc.groupby(['phone_no_m'])['voc_day_call_city_name_nunique'].transform('mean')
    df_voc.loc[(df_voc['voc_day_call_city_name_mean'] > 2) &(df_voc['voc_day_call_city_name_mean'] < 3), 'voc_day_call_city_name_mean'] = (2 + 3) / 2
    df_voc['voc_day_call_city_name_max'] = df_voc.groupby(['phone_no_m'])['voc_day_call_city_name_nunique'].transform('max')
    df_voc.loc[(df_voc['voc_day_call_city_name_max'] > 5), 'voc_day_call_city_name_max'] = 5
    df_voc['voc_day_call_city_name_min'] = df_voc.groupby(['phone_no_m'])['voc_day_call_city_name_nunique'].transform('min')
    df_voc['voc_day_call_city_name_std'] = df_voc.groupby(['phone_no_m'])['voc_day_call_city_name_nunique'].transform('std')
    df_voc.loc[(df_voc['voc_day_call_city_name_std'] > 1.5), 'voc_day_call_city_name_std'] = 1.5

    del df_voc['voc_day_call_city_name_nunique']

    df_voc['voc_day_call_county_name_nunique'] = df_voc.groupby(['phone_no_m', 'voc_day'])['county_name'].transform('nunique')
    df_voc['voc_day_call_county_name_mean'] = df_voc.groupby(['phone_no_m'])['voc_day_call_county_name_nunique'].transform('mean')
    df_voc.loc[(df_voc['voc_day_call_county_name_mean'] > 5), 'voc_day_call_county_name_mean'] = 5

    df_voc['voc_day_call_county_name_max'] = df_voc.groupby(['phone_no_m'])['voc_day_call_county_name_nunique'].transform('max')
    df_voc.loc[(df_voc['voc_day_call_county_name_max'] > 12), 'voc_day_call_county_name_max'] = 12

    df_voc['voc_day_call_county_name_min'] = df_voc.groupby(['phone_no_m'])['voc_day_call_county_name_nunique'].transform('min')
    df_voc.loc[(df_voc['voc_day_call_county_name_min'] > 3), 'voc_day_call_county_name_min'] = 3

    df_voc['voc_day_call_county_name_std'] = df_voc.groupby(['phone_no_m'])['voc_day_call_county_name_nunique'].transform('std')
    df_voc.loc[(df_voc['voc_day_call_city_name_mean'] > 3) &(df_voc['voc_day_call_city_name_mean'] < 4), 'voc_day_call_city_name_mean'] = (3 + 4) / 2

    del df_voc['voc_day_call_county_name_nunique']
    df_voc = df_voc.drop(['voc_day', 'voc_hour'], axis=1)
    # 每月
    # del df_voc['voc_day']
    # del df_voc['voc_hour']
    return df_voc


def user_voc_trend_fea(df_voc,num_workers=10):
    df_voc['start_datetime'] = pd.to_datetime(df_voc['start_datetime'])
    df_voc['voc_day'] = df_voc['start_datetime'].dt.day
    df_voc['voc_hour'] = df_voc['start_datetime'].dt.hour
    features = pd.DataFrame({'phone_no_m': df_voc['phone_no_m'].unique()})


    # 每个月的活跃天数，目前设定每日电话数量>20为活跃
    tmp = df_voc.groupby(['phone_no_m', 'voc_day'],as_index = False).agg({'opposite_no_m':'count'})
    tmp = tmp.sort_values(by=['phone_no_m', 'voc_day'], ascending=False)
    tmp['voc_judge_activate'] = 0
    tmp.loc[tmp['opposite_no_m']>20,'voc_judge_activate'] = 1
    tmp1 = tmp.groupby(['phone_no_m'],as_index = False).agg({'voc_judge_activate':'sum'})
    features = features.merge(tmp1, on='phone_no_m', how='left')


    groupby = df_voc.groupby(['phone_no_m'])

    func = partial(last_k_call_interval, periods=[50,100, 200,100000000000])
    g = parallel_apply(groupby, func, index_name='phone_no_m', num_workers=num_workers, chunk_size=10000).reset_index()
    features = features.merge(g, on='phone_no_m', how='left')

    func = partial(last_k_caller_interval, periods=[50,100, 200,10e16])
    g = parallel_apply(groupby, func, index_name='phone_no_m', num_workers=num_workers, chunk_size=10000).reset_index()
    features = features.merge(g, on='phone_no_m', how='left')



    return features