#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


# In[ ]:


TRAIN_DATA = '../input/train.csv'
TEST_DATA = '../input/test.csv'


# In[ ]:


def read_csv(path):
    logger.debug('enter')
    df = pd.read_csv(path)
    logger.debug('exit')
    return df


# In[ ]:


def load_train_data():
    logger.debug('enter')
    df = read_csv(TRAIN_DATA)
    logger.debug('exit')
    return df


# In[ ]:


def load_test_data():
    logger.debug('enter')
    df = read_csv(TEST_DATA)
    logger.debug('exit')
    return df


# In[ ]:


def gini(y, pred):
    fpr, tpr, thr = roc_curve(y, pred, pos_label=1)
    g = 2 * auc(fpr, tpr) - 1
    return g


# In[ ]:


def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', - gini(y, pred)


# In[ ]:


logger = getLogger(__name__)


# In[ ]:


DIR = 'result_tmp/'
SAMPLE_SUBMIT_FILE = '../input/sample_submission.csv'


# In[ ]:


log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s')
handler = StreamHandler()
handler.setLevel('INFO')
handler.setFormatter(log_fmt)
logger.addHandler(handler)

handler = FileHandler(DIR + 'train.py.log', 'a')
handler.setLevel(DEBUG)
handler.setFormatter(log_fmt)
logger.setLevel(DEBUG)
logger.addHandler(handler)


# In[ ]:


logger.info('start')
df = load_train_data()


# In[ ]:


cols_binary = [
    #'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_11_bin', 
    #'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin',
    'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin',
    'ps_calc_19_bin', 'ps_calc_20_bin',
    ]
cols_category = [
    'ps_ind_01', 'ps_ind_14', 'ps_ind_15', 'ps_reg_01', 'ps_reg_02', 'ps_car_15',
    'ps_ind_03', 'ps_car_11',
    #'ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat',
    #'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_05_cat', 
    #'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_10_cat', 
    #'ps_car_11_cat',
    ]
cols_numeric = [
    #'ps_ind_01', 'ps_ind_03', 'ps_ind_14', 'ps_ind_15',
    #'ps_reg_01', 'ps_reg_02', 'ps_reg_03',
    #'ps_car_11', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15',
    'ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 
    'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 
    'ps_calc_13', 'ps_calc_14',
    ]
cols_id = [
    'id',
]
cols_target = [
    'target'
]

drop_cols = cols_binary
drop_cols.extend(cols_numeric)
drop_cols.extend(cols_target)
drop_cols.extend(cols_id)
print(drop_cols)

x_train = df.drop(drop_cols, axis=1)
y_train = df['target'].values
print(x_train.head())


# In[ ]:


use_cols = x_train.columns.values

logger.info('train columns: {} {}'.format(use_cols.shape, use_cols))

logger.info('data preparation end')


# In[ ]:


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

all_params = {'max_depth': [3],
              'learning_rate': [0.1],
              'min_child_weight': [3],
              'n_estimators': [10000],
              'colsample_bytree': [0.8],
              'colsample_bylevel': [0.8],
              'reg_alpha': [0.1],
              'max_delta_step': [0.1],
              'seed': [0],
             }
min_score = 100
min_params = None

for params in tqdm(list(ParameterGrid(all_params))):
    logger.info('params: {}'.format(params))

    list_gini_score = []
    list_logloss_score = []
    list_best_iterations = []
    for train_idx, valid_idx in cv.split(x_train, y_train):
        trn_x = x_train.iloc[train_idx, :]
        val_x = x_train.iloc[valid_idx, :]

        trn_y = y_train[train_idx]
        val_y = y_train[valid_idx]

        clf = xgb.sklearn.XGBClassifier(**params)
        clf.fit(trn_x,
                trn_y,
                eval_set=[(val_x, val_y)],
                early_stopping_rounds=100,
                eval_metric=gini_xgb
                )

        pred = clf.predict_proba(val_x, ntree_limit=clf.best_ntree_limit)[:, 1]
        sc_logloss = log_loss(val_y, pred)
        sc_gini = - gini(val_y, pred)

        list_logloss_score.append(sc_logloss)
        list_gini_score.append(sc_gini)
        list_best_iterations.append(clf.best_iteration)
        logger.debug('   logloss: {}, gini: {}'.format(sc_logloss, sc_gini))
        #break
    params['n_estimators'] = int(np.mean(list_best_iterations))
    sc_logloss = np.mean(list_logloss_score)
    sc_gini = np.mean(list_gini_score)
    if min_score > sc_gini:
        min_score = sc_gini
        min_params = params
    logger.info('logloss: {}, gini: {}'.format(sc_logloss, sc_gini))
    logger.info('current min score: {}, params: {}'.format(min_score, min_params))

logger.info('minimum params: {}'.format(min_params))
logger.info('minimum gini: {}'.format(min_score))

clf = xgb.sklearn.XGBClassifier(**min_params)
clf.fit(x_train, y_train)
#with open(DIR + 'model.pkl', 'wb') as f:
#    pickle.dump(clf, f, -1)

logger.info('train end')

#with open(DIR + 'model.pkl', 'rb') as f:
#    clf = pickle.load(f)
#df = load_test_data()


# In[ ]:


df = load_test_data()

id_test = df['id'].values
x_test = df[use_cols]

logger.info('test data load end {}'.format(x_test.shape))
pred_test = clf.predict_proba(x_test)[:,1]

#df_submit = read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
df_submit=pd.DataFrame()
print(id_test.shape)
print(pred_test)
df_submit['id'] = id_test
df_submit['target'] = pred_test

df_submit.to_csv(DIR + 'submit.csv', index=False)


# In[ ]: