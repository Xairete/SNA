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
oldest = date(2018,3,21)
data_sample = parquet.read_table(input_path + '/collabTrain/date=2018-03-21').to_pandas()
dayofweek = oldest.weekday()
data_sample['dayofweek'] = str(dayofweek)
data_sample['day'] = oldest
for i in range(1,21):
    print(oldest - timedelta(i))
    day  = oldest - timedelta(i)
    dayofweek = day.weekday()
    s = '/collabTrain/date='+str((oldest - timedelta(i)))
    data1 = parquet.read_table(input_path + s).to_pandas()
    data1['dayofweek'] = str(dayofweek)
    data1['day'] = day
    data_sample = pd.concat([data_sample, data1])

feed = data_sample['feedback']
del data1

y = feed.apply(lambda x: 1.0 if("Liked" in x) else 0.0)
data_sample['liked'] = y.rename('liked').astype('Int16')

#data = data_sample.sample(frac = 0.20, random_state=546789)
data = data_sample
valid_data = data
y = data['liked']

data.info(max_cols = 172)
data.day = pd.to_datetime(data.day)
isweekend = pd.to_datetime(data.day).dt.dayofweek.apply(lambda x: 1.0 if(x==6 or x==5) else 0.0)
data['isweekend'] = isweekend
ducountlike = data[['day', 'liked']].groupby('day').sum()
ducount = data[['day', 'instanceId_userId']].groupby('day').count()
duunique = data[['day', 'instanceId_userId']].groupby('day').nunique()
dudiv = duunique['instanceId_userId'].div( ducount['instanceId_userId']) 
dulike = ducountlike['liked'].div( duunique['instanceId_userId']) 

docount = data[['day', 'instanceId_objectId']].groupby('day').count()
dounique = data[['day', 'instanceId_objectId']].groupby('day').nunique()
dodiv = dounique['instanceId_objectId'].div( docount['instanceId_objectId']) 
doudiv = dounique['instanceId_objectId'].div( duunique['instanceId_userId'])
dolike = ducountlike['liked'].div( dounique['instanceId_objectId']) 

concatdata = data_sample["instanceId_userId"].apply(str) +'_'+ data_sample["instanceId_objectId"].apply(str)
uniqdata = concatdata.unique()
from collections import Counter
nonuniquedata = Counter(concatdata)
numnonuniq = nonuniquedata.most_common(20)

nudata_sample = data_sample[data_sample['instanceId_userId'] == 9063906]

#---СОМНИТЕЛЬНАЯ ЧАСТЬ----------------------------------------------------
#User_like_count = data[['liked','instanceId_userId']].groupby('instanceId_userId').count()
#User_like_count['liked']=User_like_count['liked'].astype('Int16')
#data = data.join(User_like_count.rename(columns = {'liked':'User_like_count'}), on = 'instanceId_userId')
#
#Object_like_count = data[['liked','instanceId_objectId']].groupby('instanceId_objectId').count()
#Object_like_count['liked']=Object_like_count['liked'].astype('Int16')
#data = data.join(Object_like_count.rename(columns = {'liked':'Object_like_count'}), on = 'instanceId_objectId')

#________________________________________________________________

User_Object_count = data[['instanceId_userId','instanceId_objectId']].groupby('instanceId_userId').count().astype('Int16')
Object_User_count = data[['instanceId_userId','instanceId_objectId']].groupby('instanceId_objectId').count().astype('Int16')
User_Object_count = User_Object_count.rename(columns = {'instanceId_objectId':'User_Object_count'})
Object_User_count = Object_User_count.rename(columns = {'instanceId_userId':'Object_User_counter'})
data = data.join(User_Object_count, on = 'instanceId_userId')
data = data.join(Object_User_count, on = 'instanceId_objectId')

#Object_like_persent =Object_like_count.rename(columns = {'liked':'Like_Persent'})
#Object_like_persent['Like_Persent'] =Object_like_persent['Like_Persent'] / Object_User_count['Object_User_counter']
#data = data.join(Object_like_persent, on = 'instanceId_objectId')

data = data.drop(columns =['liked'])

data.info(max_cols=170)
data_20 = data.head(20)

missing = missing_values_table(data)
missing_columns = list(missing[missing['% of Total Values'] > 99].index)
print('We will remove %d columns.' % len(missing_columns))
data = data.drop(columns = list(missing_columns))

ids = data[['audit_experiment','metadata_options','instanceId_userId', 'instanceId_objectId', 'audit_timestamp', 'audit_timePassed']]
data = data.drop(columns = ['audit_experiment','metadata_options','feedback','instanceId_userId', 'instanceId_objectId', 'audit_timestamp', 'audit_timePassed'])
data.day = pd.to_datetime(data.day)

#data = pd.get_dummies(data)
# Fit the model and check the weight
# Read the test data
test = parquet.read_table(input_path + '/collabTest').to_pandas()
test.head(10)

#test = test.join(User_like_count.rename(columns = {'liked':'User_like_count'}), on = 'instanceId_userId')
#test = test.join(Object_like_count.rename(columns = {'liked':'Object_like_count'}), on = 'instanceId_objectId')

User_Object_count = test[['instanceId_userId','instanceId_objectId']].groupby('instanceId_userId').count().astype('Int16')
Object_User_count = test[['instanceId_userId','instanceId_objectId']].groupby('instanceId_objectId').count().astype('Int16')
User_Object_count = User_Object_count.rename(columns = {'instanceId_objectId':'User_Object_count'})
Object_User_count = Object_User_count.rename(columns = {'instanceId_userId':'Object_User_counter'})
test = test.join(User_Object_count, on = 'instanceId_userId')
test = test.join(Object_User_count, on = 'instanceId_objectId')
#test = test.join(Object_like_persent, on = 'instanceId_objectId')
test_date = pd.to_datetime(test.date)
test_days = pd.to_datetime(test.date).dt.dayofweek.apply(str)
test['dayofweek'] = test_days
test['day'] = test_date 
isweekend = pd.to_datetime(test.day).dt.dayofweek.apply(lambda x: 1.0 if(x==6 or x==5) else 0.0)
test['isweekend'] = isweekend

testducount = test[['day', 'instanceId_userId']].groupby('day').count()
testduunique = test[['day', 'instanceId_userId']].groupby('day').nunique()
testdudiv = testduunique['instanceId_userId'].div( testducount['instanceId_userId']) 

testdocount = test[['day', 'instanceId_objectId']].groupby('day').count()
testdounique = test[['day', 'instanceId_objectId']].groupby('day').nunique()
testdodiv = testdounique['instanceId_objectId'].div( testdocount['instanceId_objectId']) 
testdoudiv = testdounique['instanceId_objectId'].div( testduunique['instanceId_userId'])

test = test.drop(columns = 'date')

test_data = test.drop(columns = list(missing_columns))
ids = ['audit_experiment','metadata_options','instanceId_userId', 'instanceId_objectId', 'audit_timestamp', 'audit_timePassed']
test_data = test_data.drop(columns = ['audit_experiment','metadata_options','instanceId_userId', 'instanceId_objectId', 'audit_timestamp', 'audit_timePassed'])

#test_data = pd.get_dummies(test_data)

concattest = test["instanceId_userId"].apply(str) + test["instanceId_objectId"].apply(str)
uniq = concattest.unique()
from collections import Counter
nonunique = Counter(concattest)
nunnonuniq = nonunique.most_common(558)

print('Training Features shape: ', data.shape)
print('Testing Features shape: ', test_data.shape)
data.info(max_cols=210)
test_data.info(max_cols=210)

corr_koef = data.corr()
field_drop = [i for i in corr_koef if corr_koef[i].isnull().drop_duplicates().values[0]]
cor_field = []
for i in corr_koef:
    for j in corr_koef.index[abs(corr_koef[i]) > 0.9]:
        if i != j and j not in cor_field and i not in cor_field:
            cor_field.append(j)
            print ("%s-->%s: r^2=%f" % (i,j, corr_koef[i][corr_koef.index==j].values[0]))
            
field_drop =field_drop + cor_field

train_list = data.columns.values.tolist() 
test_list = test_data.columns.values.tolist() 
for j in test_list:
    if j not in train_list:
        print(j)
data = data.drop(field_drop, axis=1)
test_data = test_data.drop(field_drop, axis=1)
data = data.drop(columns = 'membership_status_R')
data = data.drop(columns = 'label')
#X = data.fillna(0.0)
#T = test_data.fillna(0.0)
X = data.fillna(0.0)
T = test_data.fillna(0.0)
valid_data['label'] = y
X = X.drop(columns = ['day'])
T = T.drop(columns = ['day'])
categorical_features = list(data.select_dtypes(include=['object']).columns)
categorical_features_idx = [X.columns.get_loc(c) for c in categorical_features if c in X]
import gc
gc.enable()

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=546789)

feats = [f for f in X.columns if f not in ids]

member_column_list = X.filter(regex='member', axis=1).columns.values.tolist() 
owner_column_list = X.filter(regex='owner', axis=1).columns.values.tolist()
user_column_list = X.filter(regex='user', axis=1).columns.values.tolist()
auditweights_column_list = X.filter(regex='auditweights', axis=1).columns.values.tolist()
other_column_list = [f for f in X.columns if f not in auditweights_column_list]
other_column_list = [f for f in other_column_list if f not in user_column_list]
other_column_list = [f for f in other_column_list if f not in owner_column_list]
other_column_list = [f for f in other_column_list if f not in member_column_list]
best = ['metadata_ownerId', 'metadata_authorId', 'userOwnerCounters_CREATE_LIKE', 'user_birth_date', 'objectId', 'auditweights_ctr_high', 'auditweights_matrix', 'auditweights_numLikes', 'auditweights_svd_spark', 'Object_User_counter']

max_inp = ['metadata_authorId',  'metadata_createdAt',  'metadata_numSymbols', 'Object_User_counter', 'audit_pos', 'User_Object_count',  
           'userOwnerCounters_CREATE_LIKE',  'user_birth_date',  'user_create_date',  
           'user_change_datime', 'user_ID_Location', 'auditweights_ageMs',  'auditweights_ctr_gender',  
           'auditweights_ctr_high',  'auditweights_matrix',  'auditweights_numLikes',  'auditweights_ctr_negative', 'auditweights_svd_spark']
support_feats = ['audit_pos','audit_resourceType', 'metadata_ownerId', 'metadata_createdAt', 'metadata_authorId', 
                 'metadata_numPhotos', 'metadata_numPolls', 'metadata_numSymbols', 'metadata_numVideos', 'metadata_totalVideoLength', 
                 'userOwnerCounters_USER_FEED_REMOVE', 
                 'userOwnerCounters_UNKNOWN','userOwnerCounters_CREATE_TOPIC', 'userOwnerCounters_CREATE_COMMENT', 
                 'userOwnerCounters_CREATE_LIKE', 'userOwnerCounters_TEXT', 'userOwnerCounters_IMAGE', 
                 'userOwnerCounters_VIDEO', 'membership_statusUpdateDate', 'user_create_date', 'user_birth_date', 
                 'user_gender', 'user_ID_country', 'user_ID_Location', 'user_change_datime', 
                 'user_region', 'objectId', 'auditweights_ageMs', 'auditweights_ctr_gender',
                 'auditweights_ctr_high', 'auditweights_ctr_negative',
                 'auditweights_dailyRecency', 'auditweights_feedOwner_RECOMMENDED_GROUP',
                 'auditweights_feedStats', 'auditweights_friendLikes',
                 'auditweights_likersFeedStats_hyper',
                 'auditweights_likersSvd_prelaunch_hyper', 'auditweights_matrix',
                 'auditweights_numDislikes', 'auditweights_numLikes',
                 'auditweights_numShows', 'auditweights_svd_prelaunch',
                 'auditweights_svd_spark', 'auditweights_userOwner_CREATE_COMMENT',
                 'auditweights_userOwner_CREATE_LIKE', 'auditweights_userOwner_IMAGE',
                 'auditweights_userOwner_TEXT', 'auditweights_userOwner_USER_FEED_REMOVE',
                 'auditweights_x_ActorsRelations', 'auditweights_likersSvd_spark_hyper',
                 'User_Object_count', 'Object_User_counter', 'audit_clientType_API',
                 'audit_clientType_MOB', 'audit_clientType_WEB',
                 'metadata_ownerType_GROUP_OPEN', 'metadata_platform_ANDROID',
                 'metadata_platform_OTHER', 'metadata_platform_WEB', 'membership_status_A']

#feats = data.columns.values.tolist() 
#feats = [f for f in X.columns if f in support_feats]
oof_preds = np.zeros(X.shape[0])
sub_preds = np.zeros(test.shape[0])

from sklearn.linear_model import Ridge
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool

for n_fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
        train_x, train_y = X[feats].iloc[train_idx], y.iloc[train_idx]
        val_x, val_y = X[feats].iloc[val_idx], y.iloc[val_idx]

#        clf = LGBMClassifier(
#                    boosting_type = 'gbdt', 
#                    n_estimators=1000, 
#                    learning_rate=0.1, 
#                    reg_alpha=.1, 
#                    reg_lambda=.03, 
#                    min_split_gain=.01, 
#                    min_child_weight=16, 
#                    silent=-1, 
#                    verbose=-1,
#                    random_state=546789
#                    )
#        clf.fit(train_x, train_y, 
#                    eval_set= [(train_x, train_y), (val_x, val_y)], 
#                    eval_metric='auc', verbose=100, early_stopping_rounds=30  #30
#                   )
        clf = CatBoostClassifier( 
                           n_estimators=1000,
                           learning_rate=0.2, 
                           loss_function='Logloss', 
                           logging_level='Verbose',
                           custom_metric='AUC:hints=skip_train~false', 
                           metric_period=20,
                           early_stopping_rounds=30,
                           cat_features = categorical_features_idx,
                           random_seed=546789)
        clf.fit(train_x, train_y, 
                    eval_set= [(train_x, train_y), (val_x, val_y)],
                    cat_features = categorical_features_idx,
                    verbose=100, early_stopping_rounds=30  #30
                   )
        oof_preds[val_idx] = clf.predict_proba(val_x)[:, 1]
        sub_preds -= clf.predict_proba(T[feats])[:, 1] / folds.n_splits
#        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_valid_data = valid_data[["instanceId_userId", "instanceId_objectId", 'label']].iloc[val_idx]
        sub_valid_data['score'] = oof_preds[val_idx]        
        
        sub_valid = sub_valid_data.groupby("instanceId_userId")\
            .apply(lambda y: auc(y.label.values, y.score.values))\
            .dropna().mean()
            
        print('Sub AUC : %.6f' % sub_valid)
#        sub_preds -= clf.predict_proba(T[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance = pd.DataFrame()
        fold_importance["feature"] = feats
        fold_importance["importance"] = clf.feature_importances_
        fold_importance["fold"] = n_fold + 1
        
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del  train_x, train_y, val_x, val_y, clf
        gc.collect()

print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))  


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

from sklearn.linear_model import Ridge
ridgereg = Ridge(alpha=0.1,normalize=True)



#_________________________________________________________________________
Xtimes = X[['day']]
X = X.drop(columns = ['day'])
from sklearn.feature_selection.rfe import RFECV
from lightgbm import LGBMClassifier
clf = LGBMClassifier(
                    boosting_type = 'gbdt', 
                    n_estimators=1000, 
                    learning_rate=0.1, 
                    reg_alpha=.1, 
                    reg_lambda=.03, 
                    min_split_gain=.01, 
                    min_child_weight=16, 
                    silent=-1, 
                    verbose=-1,
                    random_state=546789
                    )
selector = RFECV(clf, step=5, cv=5, verbose = 1)
selector.fit(X, y)
# summarize the selection of the attributes
print(list(selector.ranking_ )
support = np.asarray(X.columns)[selector.support_ ]
X.info()