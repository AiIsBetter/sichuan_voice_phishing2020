#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2020.06.08

import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import  StratifiedKFold,train_test_split
from sklearn.metrics import f1_score
import warnings
from utils.user_fea import user_fea
from utils.user_voc import user_voc_fea,user_voc_trend_fea

from utils.tools import display_importances
warnings.simplefilter(action='ignore', category=FutureWarning)


def score_vail(vaild_preds, real):
    """f1阈值搜索
    """
    #     import matplotlib.pylab as plt
    #     plt.figure(figsize=(16,5*10))
    best = 0
    bp = 0
    score = []
    for i in range(600):
        p = 32 + i * 0.08
        threshold_test = round(np.percentile(vaild_preds, p), 4)
        pred_int = vaild_preds > threshold_test
        ff = f1_score(pred_int, real,average = 'macro')
        score.append(ff)

        if ff >= best:
            best = ff
            bp = p
    # plt.plot(range(len(score)), score)
    # plt.show()
    return bp, best
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def main(debug = False,datetime = None):
    num_rows = 100000 if debug else None
    print('train set process.......')
    train_user = pd.read_csv('../../../data/0527/test/test_4_label.csv')
    test_user = pd.read_csv('../../../data/0722/test/test_user.csv')
    train_user['select'] = 0
    test_user['select'] = 1
    # test_user['label_copy'] = test_user['label']
    test_user['arpu_202005'] = test_user['arpu_202005'].replace({'\\N': np.nan})
    test_user['arpu_202005'] = test_user['arpu_202005'].astype(np.float64)
    test_user['label'] = -1
    df = user_fea(train_user,test_user,'arpu_202004')
    del train_user,test_user
    gc.collect()
    print('user fea finish！')
    with timer("user voc process:"):
        train_voc = pd.read_csv('../../../data/0527/test/test_voc.csv', nrows=num_rows)
        train_voc = train_voc[train_voc['start_datetime'].str.contains(datetime)]

        train_voc = train_voc.reset_index(drop=True)
        test_voc = pd.read_csv('../../../data/0722/test/test_voc.csv', nrows=num_rows)
        with timer("user voc fea process:"):
            df_voc= pd.concat([train_voc,test_voc])
            df_voc = user_voc_fea(df_voc)
            df_voc = df_voc.drop_duplicates(subset=['phone_no_m']).reset_index(drop = True)
            df_voc = df_voc.drop(['start_datetime','city_name','county_name','imei_m','opposite_no_m','calltype_id'], axis=1)
            df = df.merge(df_voc,on = 'phone_no_m',how = 'left')
            del df_voc
            gc.collect()
        with timer("user voc trend fea process:"):
            df_voc = pd.concat([train_voc, test_voc])
            del train_voc, test_voc
            gc.collect()
            df_voc = user_voc_trend_fea(df_voc,num_workers=10)
            df = df.merge(df_voc,on = 'phone_no_m',how = 'left')
            del df_voc
            gc.collect()

    # with timer("user sms process:"):
    #     train_sms = pd.read_csv('../data/0527/test/test_sms.csv', nrows=num_rows)
    #     train_sms = train_sms[train_sms['request_datetime'].str.contains(datetime)]

    #     train_sms = train_sms.reset_index(drop=True)
    #     test_sms = pd.read_csv('../data/0722/test/test_sms.csv', nrows=num_rows)
    #     with timer("user sms fea process:"):
    #         df_sms= pd.concat([train_sms,test_sms])
    #         df_sms = user_sms_fea(df_sms)
    #         df_sms = df_sms.drop_duplicates(subset=['phone_no_m']).reset_index(drop = True)
    #         df_sms = df_sms.drop(['request_datetime','opposite_no_m','calltype_id'], axis=1)
    #         df = df.merge(df_sms,on = 'phone_no_m',how = 'left')
    #         del df_sms速度快准确度高，本次只用了单模型，没有其它不同的特征集来做融合，也就没有考虑融合了，简单做了下bagging和stacking，没有提升，还是不同特征集融合应该更有效。，而且一般来说，cv好的在b榜上面成绩好的机会更大些。
    #         gc.collect()
    #     with timer("user sms trend fea process:"):
    #         df_sms= pd.concat([train_sms,test_sms])
    #         del train_sms,test_sms
    #         gc.collect()
    #         df_sms = user_sms_trend_fea(df,df_sms)
    #         df = df.merge(df_sms,on = 'phone_no_m',how = 'left')
    #         del df_sms
    #         gc.collect()
    # with timer("user app process:"):
    #     train_app = pd.read_csv('../data/0527/test/test_app.csv', nrows=num_rows)
    #     test_app = pd.read_csv('../data/0722/test/test_app.csv', nrows=num_rows)
    #     train_app = train_app[train_app['month_id'] == datetime]
    #     # tmp = tmp[~tmp['phone_no_m'].isin(train_app['phone_no_m'].unique())]
    #     # tmp['month_id'] = '2020-03'
    #     # train_app = pd.concat([train_app, tmp])
    #
    #
    #     train_app = train_app.reset_index(drop=True)
    #     with timer("user app process1:"):
    #         df_app= pd.concat([train_app,test_app])
    #         # df = df[df['label']!=-1]
    #         # tmp = df[~df['phone_no_m'].isin(df_app['phone_no_m'].unique())]
    #         # a = tmp['label'].sum()
    #
    #         df_app = user_app_fea(df_app,debug)
    #         df_app = df_app.drop_duplicates(subset=['phone_no_m']).reset_index(drop = True)
    #         df_app = df_app.drop(['month_id','busi_name'],axis = 1)
    #         df = df.merge(df_app,on = 'phone_no_m',how = 'left')
    #         del df_app
    #         gc.collect()
    #     with timer("user app process2:"):
    #         df_app = pd.concat([train_app, test_app])
    #         del train_app, test_app
    #         gc.collect()
    #         df_app = user_app_other_fea(df,df_app)
    #         df = df.merge(df_app, on='phone_no_m', how='left')
    #         del df_app
    #         gc.collect()
    # 绘图
    #     train_user = df[df['label'] != -1]
    #     test_user = df[df['label'] == -1]
    #     for i in train_user.drop(['phone_no_m', 'label','select'], axis=1).columns:
    #         print(i)
    #         plt.figure()
    #         plt.subplot(1, 2, 1)
    #         sns.distplot(train_user[i], hist=False)
    #         plt.title(i)
    #         plt.xlabel('train_user')
    #         plt.subplot(1, 2, 2)
    #         sns.distplot(test_user[i], hist=False)
    #         plt.title(i)
    #         plt.xlabel('test_user')
    #         # plt.show()
    #         plt.savefig('pic/{}.jpg'.format(i))
    #         plt.close()
    with timer("train data select"):
        df = df.fillna(-1)
        df_train,df_test,df_tr_y,df_te_y = train_test_split(df.drop('select',axis = 1),df['select'],
                                                            random_state = 2020,test_size = 0.2,
                                                            stratify = df['select'])
        train_columns = [i for i in df_train.columns if i not in ['phone_no_m', 'label','select']]
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 170,
            'max_depth': -1,
            'learning_rate': 0.05,
            'max_bin': 301,
            'subsample_for_bin': 299937,
            "feature_fraction": 0.843,
            "bagging_fraction": 0.866,
            "bagging_freq": 7,
            "min_child_samples": 89,
            'lambda_l1': 0.006237830242067111,
            'lambda_l2': 2.016472023736186e-05,
            'random_state': 2020,
            'verbose': -1,
        }
        scores = 0
        threshold = 0
        print('start training......')
        tr_x = lgb.Dataset(df_train[train_columns], label=df_tr_y)
        te_x = lgb.Dataset(df_test[train_columns], label=df_te_y)
        clf = lgb.train(params,
                        tr_x,
                        num_boost_round=5000,
                        valid_sets=(tr_x, te_x),
                        valid_names=('train', 'validate'),
                        early_stopping_rounds=100, verbose_eval=100)
        pred = clf.predict(df_test[train_columns], num_iteration=clf.best_iteration)
        print('auc score....: ',roc_auc_score(df_te_y,pred))

    with timer("model train"):
        train_user = df[df['label']!=-1]
        test_user = df[df['label']==-1]
        # 训练参数
        folds = 5
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=2020)
        train_mean = np.zeros(shape=[1,folds])
        predict = np.zeros(shape=[test_user.shape[0], folds],dtype=float)
        train_columns = [i for i in train_user.columns if i not in ['phone_no_m', 'label']]
        feature_importance_df = pd.DataFrame()

        def lgb_f1_score(y_hat, data):
            y_true = data.get_label()
            y_hat = np.round(y_hat)  # scikits f1 doesn't like probabilities
            return 'f1', f1_score(y_true, y_hat,average = 'macro'), True
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            # 'metric': 'auc',
            'num_leaves': 170,
            'max_depth': -1,
            'learning_rate': 0.05,
            'max_bin': 301,
            'subsample_for_bin': 299937,
            "feature_fraction": 0.843,
            "bagging_fraction": 0.866,
            "bagging_freq": 7,
            "min_child_samples": 89,
            'lambda_l1': 0.006237830242067111,
            'lambda_l2': 2.016472023736186e-05,
            'random_state': 2020,
            'verbose': -1,
        }
        scores  = 0
        threshold = 0
        print('start training......')
        for i, (trn_idx, val_idx) in enumerate(skf.split(train_user[train_columns], train_user['label'])):
            tr_x = train_user[train_columns].iloc[trn_idx]
            tr_y = train_user['label'].iloc[trn_idx]
            te_x = train_user[train_columns].iloc[val_idx]
            te_y = train_user['label'].iloc[val_idx]

            tr_x = lgb.Dataset(tr_x, label=tr_y)
            te_x = lgb.Dataset(te_x, label=te_y)
            clf = lgb.train(params,
                            tr_x,
                            num_boost_round=5000,
                            valid_sets=(tr_x, te_x),
                            valid_names=('train', 'validate'),
                            early_stopping_rounds=100,verbose_eval =100,feval=lgb_f1_score)


            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = train_columns
            fold_importance_df["importance"] = clf.feature_importance()
            fold_importance_df["fold"] = i + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

            sub = clf.predict(train_user[train_columns].iloc[val_idx],num_iteration=clf.best_iteration)

            th, bestf1 = score_vail(sub, train_user['label'].iloc[val_idx])
            print("score: ", bestf1,"threshold: ", th)
            scores += bestf1
            threshold += th
            predict[:, i] = clf.predict(test_user[train_columns],num_iteration=clf.best_iteration)
            train_mean[0, i] = bestf1
            print('Train set kfold {} f1 accuracy :'.format(i),bestf1)
        display_importances(feature_importance_df)
        print('Train set mean f1 accuracy :'.format(i), np.mean(train_mean))
        threshold = threshold / folds
        test_p = np.array(predict)
        test_p = test_p.mean(axis=1)
        test_user["prob"] = test_p
        test_user["label"] = test_user["prob"] > round(np.percentile(test_user["prob"], threshold), 4)
        test_user["label"] = test_user["label"].astype(int)
        test_user[['phone_no_m','label']].to_csv('submission_{}.csv'.format(datetime), index=False)

if __name__ == "__main__":
    with timer("training......"):
        main(debug=False,datetime = '2020-04')