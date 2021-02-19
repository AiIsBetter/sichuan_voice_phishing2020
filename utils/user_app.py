#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2020.06.08

import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import math
def user_app_fea(df_app,debug):
    df_app['busi_name'] = df_app['busi_name'].fillna('-1')
    df_app['busi_name'] = df_app['busi_name'].replace({'-1': 'un'})
    df_app = df_app.sort_values(by=['phone_no_m', 'flow'], ascending=False)
    df_app = df_app.drop_duplicates(subset=['phone_no_m', 'busi_name']).reset_index(drop=True)

    df_app['total_flow'] = df_app.groupby('phone_no_m')['flow'].transform('sum')
    df_app['flow_max'] = df_app.groupby('phone_no_m')['flow'].transform('max')
    df_app['flow_min'] = df_app.groupby('phone_no_m')['flow'].transform('min')
    df_app['flow_mean'] = df_app.groupby('phone_no_m')['flow'].transform('mean')
    df_app['flow_std'] = df_app.groupby('phone_no_m')['flow'].transform('std')

    # 应用数Q
    df_app['app_count'] = df_app.groupby('phone_no_m')['phone_no_m'].transform('count')
    df_app['app_size'] = df_app.groupby('phone_no_m')['phone_no_m'].transform('size')
    df_app['app_ratio'] = df_app['app_count']/df_app['app_size']

    df_app = df_app.drop('flow',axis = 1)

    return df_app

def user_app_other_fea(df,df_app,num_workers = 10):
    features = pd.DataFrame({'phone_no_m': df_app['phone_no_m'].unique()})

    df_app['busi_name'] = df_app['busi_name'].fillna('-1')
    df_app['busi_name'] = df_app['busi_name'].replace({'-1': 'un'})

    df_app = df_app.sort_values(by = ['phone_no_m','flow'],ascending=False)
    df_app = df_app.drop_duplicates(subset=['phone_no_m','busi_name']).reset_index(drop = True)


    tmp = df_app.groupby('phone_no_m')['busi_name'].apply(lambda x: x.str.cat(sep=',')).reset_index()
    tmp['busi_name'] = tmp['busi_name'].apply(lambda x: x.split(','))
    tmp['app_length'] = tmp['busi_name'].apply(lambda x: len(x))
    features = features.merge(tmp[['phone_no_m','app_length']], on='phone_no_m', how='left')
    df_app['flow'] = df_app['flow'].apply(lambda x: int(math.log(1 + x * x*x)))
    df_app['busi_name_tmp'] = df_app['busi_name']
    lbl = LabelEncoder()
    df_app['busi_name'] = lbl.fit_transform(df_app['busi_name'].astype(str))
    busi_people = df_app.groupby('busi_name', as_index=False)['phone_no_m'].count()
    busi_people = busi_people.sort_values(by=['phone_no_m'], ascending=False)
    busi_people = busi_people[busi_people['phone_no_m'] > 300]
    #
    features['busi_name'] = 1
    for name in tqdm(busi_people['busi_name'].unique()):
        features['busi_name'] = name
        features = features.merge(df_app[['busi_name', 'phone_no_m', 'flow']], on=['phone_no_m', 'busi_name'],
                                  how='left')
        features = features.rename(columns={'flow': 'app_flow_{}'.format(name)})

    features = features.drop('busi_name', axis=1)

    return features