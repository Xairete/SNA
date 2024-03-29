# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 14:26:42 2019

@author: gev
"""
# Used to read the Parquet data
import pyarrow.parquet as parquet
# Used to train the baseline model
import numpy as np
import pandas as pd
# Where the downloaded data are
input_path = 'e:/Other/Projects/MLBC/SNAHackathon2019/'
# Where to store results
output_path = 'e:/Other/Projects/MLBC/SNAHackathon2019/'
pd.set_option('display.max_columns', None)
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

def auc(labels, scores):
    if len(labels) > sum(labels) > 0:
        return roc_auc_score(labels, scores)

    return float('NaN')

from datetime import date, timedelta
oldest = date(2018,3,20)
data = parquet.read_table(input_path + '/collabTrain/date=2018-03-21').to_pandas()

for i in range(18):
    print(oldest - timedelta(i))
    s = '/collabTrain/date='+str((oldest - timedelta(i)))
    data1 = parquet.read_table(input_path + s).to_pandas()
    day = oldest - timedelta(i)
    dayofweek = day.weekday()
    data1['dayofweek'] = str(dayofweek)
    data = pd.concat([data, data1])

feed = data['feedback']
data = data.drop(columns = ['feedback'])
del data1

y = feed.apply(lambda x: 1.0 if("Liked" in x) else 0.0)
data['liked'] = y.rename('liked').astype('Int16')

#data_sample = data.sample(frac = 0.1, random_state = 10)
data_sample = data
valid_data = data_sample

#---СОМНИТЕЛЬНАЯ ЧАСТЬ----------------------------------------------------

User_like_count = data_sample[['liked','instanceId_userId']].groupby('instanceId_userId').sum()
User_like_count['liked']=User_like_count['liked'].astype('Int16')
#data_sample = data_sample.join(User_like_count.rename(columns = {'liked':'User_like_count'}), on = 'instanceId_userId')

Object_like_count = data_sample[['liked','instanceId_objectId']].groupby('instanceId_objectId').sum()
Object_like_count['liked']=Object_like_count['liked'].astype('Int16')
#data_sample = data_sample.join(Object_like_count.rename(columns = {'liked':'Object_like_count'}), on = 'instanceId_objectId')

#________________________________________________________________

User_Object_count = data_sample[['instanceId_userId','instanceId_objectId']].groupby('instanceId_userId').count().astype('Int16')
Object_User_count = data_sample[['instanceId_userId','instanceId_objectId']].groupby('instanceId_objectId').count().astype('Int16')
User_Object_count = User_Object_count.rename(columns = {'instanceId_objectId':'User_Object_count'})
Object_User_count = Object_User_count.rename(columns = {'instanceId_userId':'Object_User_counter'})
data_sample = data_sample.join(User_Object_count, on = 'instanceId_userId')
data_sample = data_sample.join(Object_User_count, on = 'instanceId_objectId')

#Object_like_persent =Object_like_count.rename(columns = {'liked':'Like_Persent'})
#Object_like_persent['Like_Persent'] =(Object_like_persent['Like_Persent'] / Object_User_count['Object_User_counter'])



#data_sample = data_sample.join(Object_like_persent, on = 'instanceId_objectId')

y_sample = data_sample['liked']
data_sample = data_sample.drop(columns =['liked'])


missing = missing_values_table(data_sample)
missing_columns = list(missing[missing['% of Total Values'] > 95].index)
print('We will remove %d columns.' % len(missing_columns))
data_sample = data_sample.drop(columns = list(missing_columns))

ids = ['audit_experiment','metadata_options','instanceId_userId', 'instanceId_objectId', 'audit_timestamp', 'audit_timePassed', 'objectId']
data_sample = data_sample.drop(columns = ids)

data_sample = pd.get_dummies(data_sample)
# Fit the model and check the weight
# Read the test data
test = parquet.read_table(input_path + '/collabTest').to_pandas()
test_10 = test.head(10)
test_days = pd.to_datetime(test.date).dt.dayofweek
test_days = test_days.apply(str)
test['dayofweek'] = test_days
test = test.drop(columns = 'date')

#test = test.join(User_like_count.rename(columns = {'liked':'User_like_count'}), on = 'instanceId_userId')
#test = test.join(Object_like_count.rename(columns = {'liked':'Object_like_count'}), on = 'instanceId_objectId')

User_Object_count = test[['instanceId_userId','instanceId_objectId']].groupby('instanceId_userId').count().astype('Int16')
Object_User_count = test[['instanceId_userId','instanceId_objectId']].groupby('instanceId_objectId').count().astype('Int16')
User_Object_count = User_Object_count.rename(columns = {'instanceId_objectId':'User_Object_count'})
Object_User_count = Object_User_count.rename(columns = {'instanceId_userId':'Object_User_counter'})
test = test.join(User_Object_count, on = 'instanceId_userId')
test = test.join(Object_User_count, on = 'instanceId_objectId')
#test = test.join(Object_like_persent, on = 'instanceId_objectId')

test_data = test.drop(columns = list(missing_columns))

test_data = test_data.drop(columns = ids)
test_data = pd.get_dummies(test_data)

print('Training Features shape: ', data_sample.shape)
print('Testing Features shape: ', test_data.shape)
data_sample.info(max_cols=210)
test_data.info(max_cols=210)

corr_koef = data_sample.corr()
field_drop = [i for i in corr_koef if corr_koef[i].isnull().drop_duplicates().values[0]]
cor_field = []
for i in corr_koef:
    for j in corr_koef.index[abs(corr_koef[i]) > 0.9]:
        if i != j and j not in cor_field and i not in cor_field:
            cor_field.append(j)
            print ("%s-->%s: r^2=%f" % (i,j, corr_koef[i][corr_koef.index==j].values[0]))
            
field_drop =field_drop + cor_field

train_list = data_sample.columns.values.tolist() 
test_list = test_data.columns.values.tolist() 
for j in test_list:
    if j not in train_list:
        print(j)
data_sample = data_sample.drop(field_drop, axis=1)
test_data = test_data.drop(field_drop, axis=1)

data_sample, test_data = data_sample.align(test_data, join = 'inner', axis = 1)

X = data_sample.fillna(0.0)
test_data = test_data.fillna(0.0)
 
import gc
gc.enable()

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=546789)

oof_preds = np.zeros(X.shape[0])
sub_preds = np.zeros(test.shape[0])

feats = [f for f in X.columns if f not in ids]

from lightgbm import LGBMClassifier
for n_fold, (train_idx, val_idx) in enumerate(folds.split(X, y_sample)):
        train_x, train_y = X[feats].iloc[train_idx], y_sample.iloc[train_idx]
        val_x, val_y = X[feats].iloc[val_idx], y_sample.iloc[val_idx]
        
        clf = LGBMClassifier(
                    boosting_type = 'gbdt', 
                    n_estimators=1000, 
                    learning_rate=0.1, 
                    #num_leaves=8, 
                    #colsample_bytree=0.2, 
                    #subsample=0.01, 
                    #max_depth=8, 
                    reg_alpha=.1, 
                    reg_lambda=.03, 
                    min_split_gain=.01, 
                    min_child_weight=16, 
                    silent=-1, 
                    verbose=-1,
                    random_state=546789
                    )
            
        clf.fit(train_x, train_y, 
                    eval_set= [(train_x, train_y), (val_x, val_y)], 
                    eval_metric='auc', verbose=100, early_stopping_rounds=30  #30
                   )
        
        
        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds -= clf.predict_proba(test_data[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
        fold_importance = pd.DataFrame()
        fold_importance["feature"] = feats
        fold_importance["importance"] = clf.feature_importances_
        fold_importance["fold"] = n_fold + 1
        
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, train_x, train_y, val_x, val_y
        gc.collect()

print('Full AUC score %.6f' % roc_auc_score(y_sample, oof_preds))  
#
valid_data['label'] = y_sample
valid_data['score'] = oof_preds

valid = valid_data.groupby("instanceId_userId")\
    .apply(lambda y: auc(y.label.values, y.score.values))\
    .dropna().mean()
    
    
test["predictions"] = sub_preds 
result = test[["instanceId_userId", "instanceId_objectId", "predictions"]].sort_values(
    by=['instanceId_userId', 'predictions'])
result.head(10)    
# Collect predictions for each user
submit = result.groupby("instanceId_userId")['instanceId_objectId'].apply(list)
submit.head(10)
# Persist the first submit
submit.to_csv(output_path + "/collabSubmit.csv.gz", header = False, compression='gzip')        
