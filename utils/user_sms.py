#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2020.06.08

import pandas as pd
from functools import partial
from utils.tools import parallel_apply
from utils.feature_extraction import add_features_in_group

def last_k_sms_interval(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['request_datetime'], inplace=True)
    gr_['sms_interval'] = gr_['request_datetime'].diff()
    gr_['sms_interval'] = gr_['sms_interval'].dt.total_seconds()
    gr_['sms_interval'] = gr_['sms_interval'].fillna(0)
    features = {}
    for period in periods:
        if period > 10e10:
            period_name = 'sms_all'
            gr_period = gr_.copy()
        else:
            period_name = 'sms_last_{}_'.format(period)
            gr_period = gr_.iloc[:period]

        features = add_features_in_group(features, gr_period, 'sms_interval',
                                         ['mean','sum', 'max', 'min', 'std'],
                                         # ['diff'],
                                         period_name)
    return features
def last_k_smser_interval(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['request_datetime'], inplace=True)
    gr_=gr_[gr_['calltype_id']==1]
    gr_['smser_interval'] = gr_['request_datetime'].diff()
    gr_['smser_interval'] = gr_['smser_interval'].dt.total_seconds()
    gr_['smser_interval'] = gr_['smser_interval'].fillna(0)
    features = {}
    for period in periods:
        if period > 10e10:
            period_name = 'sms_all'
            gr_period = gr_.copy()
        else:
            period_name = 'sms_last_{}_'.format(period)
            gr_period = gr_.iloc[:period]

        features = add_features_in_group(features, gr_period, 'smser_interval',
                                         ['mean','sum', 'max', 'min', 'std'],
                                         # ['diff'],
                                         period_name)
    return features

def user_sms_fea(df_sms):
    df_sms['request_datetime'] = pd.to_datetime(df_sms['request_datetime'])

    df_sms['sms_day'] = df_sms['request_datetime'].dt.day
    df_sms['sms_hour'] = df_sms['request_datetime'].dt.hour
    df_sms['sms_day_mean'] = df_sms.groupby(['phone_no_m'])['sms_day'].transform('mean')
    df_sms['sms_hour_mean'] = df_sms.groupby(['phone_no_m'])['sms_hour'].transform('mean')
    # df_sms['sms_ts'] = pd.to_timedelta(df_sms['request_datetime'], unit='ns').dt.total_seconds().astype(int)
    # df_sms = df_sms.sort_values(by='sms_ts').reset_index(drop=True)
    # 每月内短信统计量
    df_sms['sms_day_count'] = df_sms.groupby(['phone_no_m'])['phone_no_m'].transform('count')
    # 每天内短信统计量
    df_sms['sms_day_count'] = df_sms.groupby(['phone_no_m', 'sms_day'])['phone_no_m'].transform('count')
    df_sms['sms_day_count_max'] = df_sms.groupby('phone_no_m')['sms_day_count'].transform('max')
    df_sms['sms_day_count_min'] = df_sms.groupby('phone_no_m')['sms_day_count'].transform('min')
    df_sms['sms_day_count_mean'] = df_sms.groupby('phone_no_m')['sms_day_count'].transform('mean')
    df_sms['sms_day_count_std'] = df_sms.groupby('phone_no_m')['sms_day_count'].transform('std')
    del df_sms['sms_day_count']
    # 每小时内短信统计量
    df_sms['sms_hour_count'] = df_sms.groupby(['phone_no_m', 'sms_hour'])['phone_no_m'].transform('count')
    df_sms['sms_hour_count_max'] = df_sms.groupby('phone_no_m')['sms_hour_count'].transform('max')
    df_sms['sms_hour_count_min'] = df_sms.groupby('phone_no_m')['sms_hour_count'].transform('min')
    df_sms['sms_hour_count_mean'] = df_sms.groupby('phone_no_m')['sms_hour_count'].transform('mean')
    df_sms['sms_hour_count_std'] = df_sms.groupby('phone_no_m')['sms_hour_count'].transform('std')
    del df_sms['sms_hour_count']


    df_sms = df_sms.drop(['sms_day', 'sms_hour'], axis=1)

    return df_sms

def user_sms_trend_fea(df,df_sms, num_workers=10):
    df_sms['request_datetime'] = pd.to_datetime(df_sms['request_datetime'])
    features = pd.DataFrame({'phone_no_m': df_sms['phone_no_m'].unique()})

    df_sms['sms_day'] = df_sms['request_datetime'].dt.day

    # 上下行短信比例以及单独统计
    tmp = df_sms.groupby(['phone_no_m','calltype_id'],as_index = False)['calltype_id'].agg({'sms_calltype_id1_count':'count'})
    features['calltype_id'] = 1
    features = features.merge(tmp,on = ['phone_no_m','calltype_id'],how = 'left')
    tmp = tmp.rename(columns = {'sms_calltype_id1_count':'sms_calltype_id2_count'})
    features['calltype_id'] = 2
    features = features.merge(tmp,on = ['phone_no_m','calltype_id'],how = 'left')
    features['sms_calltype_ratio'] =features['sms_calltype_id1_count']/(features['sms_calltype_id1_count']+features['sms_calltype_id2_count'])
    features['sms_calltype_judge'] = (features['sms_calltype_id1_count']>features['sms_calltype_id2_count']).astype(int)
    features = features.drop('calltype_id',axis = 1)
    groupby = df_sms.groupby(['phone_no_m'])
    func = partial(last_k_sms_interval, periods=[50,100, 200, 500, 10e16])

    g = parallel_apply(groupby, func, index_name='phone_no_m', num_workers=num_workers, chunk_size=10000).reset_index()
    features = features.merge(g, on='phone_no_m', how='left')

    func = partial(last_k_smser_interval, periods=[50,100,10e16])
    # 500, 800, 1500, 10e16
    g = parallel_apply(groupby, func, index_name='phone_no_m', num_workers=num_workers, chunk_size=10000).reset_index()
    features = features.merge(g, on='phone_no_m', how='left')


    return features


