#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2020.06.08

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def user_fea(train_user,test_user,date_time):

    train_user = train_user[['phone_no_m', 'city_name', 'county_name', 'idcard_cnt', date_time, 'label']]
    train_user = train_user.rename(columns={date_time: 'arpu_202005'})
    df = pd.concat([train_user,test_user],axis = 0)
    df.loc[df['arpu_202005']>800,'arpu_202005'] = 800
    df = df.reset_index(drop = True)


    df['idcard_ratio'] = df['idcard_cnt']/df['arpu_202005']

    df['ga'] = 0
    df.loc[df['city_name']=='广安','ga'] = 1
    df['tfxq'] = 0
    df.loc[df['city_name'] == '天府新区', 'tfxq'] = 1
    df['dy'] = 0
    df.loc[df['city_name'] == '德阳', 'dy'] = 1
    df['cd'] = 0
    df.loc[df['city_name'] == '成都', 'cd'] = 1
    df['bz'] = 0
    df.loc[df['city_name'] == '巴中', 'bz'] = 1
    df['ls'] = 0
    df.loc[df['city_name'] == '乐山', 'ls'] = 1

    for f in tqdm(['city_name', 'county_name']):
        lbl = LabelEncoder()
        df[f] = df[f].fillna('NA')
        df[f] = lbl.fit_transform(df[f].astype(str))

    df['user_rich'] = (df['arpu_202005'] > 500).astype(int)

    return df