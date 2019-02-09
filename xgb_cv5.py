#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#%matplotlib inline
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, ParameterGrid
from tqdm import tqdm
from logging import getLogger, StreamHandler, DEBUG, Formatter, FileHandler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import pickle


# In[ ]:


TRAIN_DATA = '../input/train.csv'
TEST_DATA = '../input/test.csv'


# In[ ]:


def read_csv(path):
    logger.debug('enter')
    df = pd.read_csv(path, parse_dates=[0])
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
test = load_test_data()


# In[ ]:


df.head()


# In[ ]:


print(df.shape)
print(test.shape)


# In[ ]:


df['count'] = np.log(df['count'] + 1)


# In[ ]:


df.head()


# In[ ]:


df.rename(columns={'count':'rentals'}, inplace=True)


# In[ ]:


df.head()


# In[ ]:


test.head()


# In[ ]:


df = df.append(test,sort=False)


# In[ ]:


df.head()


# In[ ]:


print(df.shape)


# In[ ]:


df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['dayofweek'] = df['datetime'].dt.dayofweek
df['hour'] = df['datetime'].dt.hour


# In[ ]:


df.head()


# In[ ]:


df.sort_values('datetime', inplace=True)


# In[ ]:


test = df[df['rentals'].isnull()]
df = df[~df['rentals'].isnull()]


# In[ ]:


print(df.shape)
print(test.shape)


# In[ ]:


removed_cols = ['rentals', 'casual', 'registered', 'datetime', 'atemp', 'holiday', 'month']


# In[ ]:


feats = [c for c in df.columns if c not in removed_cols]


# In[ ]:


def logmse(y, pred):
    g = mean_squared_error(y, pred)**(1/2)
    return g


# In[ ]:


logger = getLogger(__name__)


# In[ ]:


kf = KFold(n_splits=5, shuffle=True, random_state=0)

all_params = {'max_depth': [3, 5, 7],
              'learning_rate':[0.1],
              'min_child_weight':[1, 3, 5, 7],
             'n_estimators':[10000],
              'colsample_bytree': [0.8, 0.9, 1],
              'colsample_bylevel':[0.8, 0.9, 1],
              'reg_alpha':[0, 0.1],
              'max_delta_step':[0, 0.1],
            'n_jobs':[-1],
            'random_state':[0],
             'seed':[0]}

x_train = df[feats]
y_train = df['rentals'].values


min_score = 100
min_params = None

for params in tqdm(list(ParameterGrid(all_params))):
    logger.info('params: {}'.format(params))
    
    list_logmse_score = []
    list_best_iterations = []


    for train_idx, valid_idx in kf.split(x_train, y_train):
        trn_x = x_train.iloc[train_idx, :]
        val_x = x_train.iloc[valid_idx, :]
    
        trn_y = y_train[train_idx]
        val_y = y_train[valid_idx]
    
        clf = xgb.sklearn.XGBRegressor(**params)
        clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=100, eval_metric="rmse")
        y_pred = clf.predict(val_x)
    
        sc_logmse = logmse(val_y, y_pred)
    
        list_logmse_score.append(sc_logmse)
        list_best_iterations.append(clf.best_iteration)
        logger.debug('logmse:{}'.format(sc_logmse))

    params['n_estimators'] = int(np.mean(list_best_iterations))
    sc_logmse = np.mean(list_logmse_score)
    if min_score > sc_logmse:
        min_score = sc_logmse
        min_params = params

logger.info('minimam params:{}'.format(min_params))
logger.info('minimam logmse:{}'.format(min_score))
print('minimam logmse:{}',min_score)
clf = xgb.sklearn.XGBRegressor(**min_params)
clf.fit(x_train, y_train)

logger.info('train end')


# In[ ]:


with open(DIR + 'xgb_gs_model.pkl', 'wb') as f:
    pickle.dump(clf, f, -1)

with open(DIR + 'xgb_gs_model.pkl', 'rb') as f:
    clf = pickle.load(f)


# In[ ]:


test['count'] = np.exp(clf.predict(test[feats]))


# In[ ]:


test[['datetime', 'count']].to_csv(DIR + 'xgb_gs.csv', index=False)


# In[ ]:



