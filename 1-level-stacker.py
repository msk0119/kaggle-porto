#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


# Preprocessing 
id_test = test['id'].values
target_train = train['target'].values

train = train.drop(['target','id'], axis = 1)
test = test.drop(['id'], axis = 1)

col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(col_to_drop, axis=1)  
test = test.drop(col_to_drop, axis=1)  

train = train.replace(-1, np.nan)
test = test.replace(-1, np.nan)

cat_features = [a for a in train.columns if a.endswith('cat')]

for column in cat_features:
    temp = pd.get_dummies(pd.Series(train[column]))
    train = pd.concat([train,temp],axis=1)
    train = train.drop([column],axis=1)
    
for column in cat_features:
    temp = pd.get_dummies(pd.Series(test[column]))
    test = pd.concat([test,temp],axis=1)
    test = test.drop([column],axis=1)


print(train.values.shape, test.values.shape)


# In[ ]:


class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=0).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
#                y_holdout = y[test_idx]

                print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
                clf.fit(X_train, y_train)
#                cross_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
#                print("    cross_score: %.5f" % (cross_score.mean()))
                y_pred = clf.predict_proba(X_holdout)[:,1]                

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:,1]
            S_test[:, i] = S_test_i.mean(axis=1)

        results = cross_val_score(self.stacker, S_train, y, cv=3, scoring='roc_auc')
        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict_proba(S_test)[:,1]
        return res


# In[ ]:


# XGBoost params
xgb_params = {}
xgb_params['objective'] = 'binary:logistic'
xgb_params['booster'] = 'gbtree'
xgb_params['learning_rate'] = 0.1
xgb_params['n_estimators'] = 1000
xgb_params['max_depth'] = 3
xgb_params['subsample'] = 0.8
xgb_params['colsample_bytree'] = 0.8
xgb_params['min_child_weight'] = 5
xgb_params['base_score'] = 0.5
xgb_params['colsample_bylevel'] = 1
xgb_params['gamma'] = 10
xgb_params['max_delta_step'] = 0
xgb_params['min_child_weight'] = 5
xgb_params['random_state'] = 0
xgb_params['reg_alpha'] = 8
xgb_params['reg_lambda'] = 1.3
xgb_params['scale_pos_weight'] = 1.6
xgb_params['seed'] = 0


# In[ ]:


# LightGBM params
lgb_params = {}
lgb_params['learning_rate'] = 0.1
lgb_params['n_estimators'] = 139
lgb_params['max_bin'] = 10
lgb_params['subsample'] = 0.7
lgb_params['colsample_bytree'] = 0.8   
lgb_params['min_child_samples'] = 10
lgb_params['seed'] = 0
lgb_params['min_child_weight'] = 150
lgb_params['num_levels'] = 16


# In[ ]:


xgb_model = XGBClassifier(**xgb_params)

lgb_model = LGBMClassifier(**lgb_params)

log_model = LogisticRegression()


# In[ ]:


stack = Ensemble(n_splits=3,
        stacker = log_model,
        base_models = (lgb_model, xgb_model))  


# In[ ]:


y_pred = stack.fit_predict(train, target_train, test) 


# In[ ]:


sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_pred
sub.to_csv('stacked_1.csv', index=False)


# In[ ]:



