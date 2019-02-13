# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 14:26:42 2019

@author: gev
"""
# Used to read the Parquet data
import pyarrow.parquet as parquet
# Used to train the baseline model
from sklearn.linear_model import LogisticRegression
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



data = parquet.read_table(input_path + '/collabTrain/date=2018-02-03').to_pandas()
data1 = parquet.read_table(input_path + '/collabTrain/date=2018-03-03').to_pandas()
data2 = parquet.read_table(input_path + '/collabTrain/date=2018-02-16').to_pandas()
data3 = parquet.read_table(input_path + '/collabTrain/date=2018-03-12').to_pandas()
data4 = parquet.read_table(input_path + '/collabTrain/date=2018-02-18').to_pandas()
data5 = parquet.read_table(input_path + '/collabTrain/date=2018-03-18').to_pandas()
data6 = parquet.read_table(input_path + '/collabTrain/date=2018-03-19').to_pandas()
data7 = parquet.read_table(input_path + '/collabTrain/date=2018-03-20').to_pandas()
data8 = parquet.read_table(input_path + '/collabTrain/date=2018-03-21').to_pandas()
data = pd.concat([data, data1, data2, data3, data4, data5, data6, data7, data8])

feed = data['feedback']
options = data['metadata_options']
feed.head(10)
del [data1, data2, data3, data4, data5, data6, data7]
data.info(max_cols=170)
data_10 = data.head(20)
# Construct the label (liked objects)
y = feed.apply(lambda x: 1.0 if("Liked" in x and not ("Disliked" in x)) else 0.0)

missing = missing_values_table(data)
missing_columns = list(missing[missing['% of Total Values'] > 99].index)
print('We will remove %d columns.' % len(missing_columns))
data = data.drop(columns = list(missing_columns))
data = data.drop(columns = 'metadata_options')

ids = data[['instanceId_userId', 'instanceId_objectId', 'audit_timestamp', 'audit_timePassed']]
data = data.drop(columns = ['instanceId_userId', 'instanceId_objectId', 'audit_timestamp', 'audit_timePassed'])
data = data.drop(columns = ['feedback'])

data = pd.get_dummies(data)
# Fit the model and check the weight
# Read the test data
test = parquet.read_table(input_path + '/collabTest').to_pandas()
test.head(10)

test_data = test.drop(columns = list(missing_columns))
test_data = test_data.drop(columns = 'metadata_options')
test_data = test_data.drop(columns = ['instanceId_userId', 'instanceId_objectId', 'audit_timestamp', 'audit_timePassed'])
test_data = test_data.drop(columns = 'date')
test_data = pd.get_dummies(test_data)

print('Training Features shape: ', data.shape)
print('Testing Features shape: ', test.shape)
test.info(max_cols=210)

corr_koef = data.corr()
field_drop = [i for i in corr_koef if corr_koef[i].isnull().drop_duplicates().values[0]]
cor_field = []
for i in corr_koef:
    for j in corr_koef.index[abs(corr_koef[i]) > 0.9]:
        if i != j and j not in cor_field and i not in cor_field:
            cor_field.append(j)
            print ("%s-->%s: r^2=%f" % (i,j, corr_koef[i][corr_koef.index==j].values[0]))
            
#field_drop =field_drop + cor_field
field_drop = cor_field
train_list = data.columns.values.tolist() 
test_list = test_data.columns.values.tolist() 
for j in test_list:
    if j not in train_list:
        print(j)
data = data.drop(field_drop, axis=1)
test_data = test_data.drop(field_drop, axis=1)

X = data.fillna(0.0)
import gc
gc.enable()
data.columns.values.tolist()
data.info()
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=546789)

oof_preds = np.zeros(X.shape[0])
sub_preds = np.zeros(test.shape[0])

feats = [f for f in X.columns if f not in ids]

from lightgbm import LGBMClassifier
for n_fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
        train_x, train_y = X[feats].iloc[train_idx], y.iloc[train_idx]
        val_x, val_y = X[feats].iloc[val_idx], y.iloc[val_idx]
        
        clf = LGBMClassifier(
                    boosting_type = 'gbdt', 
                    n_estimators=1000, 
                    learning_rate=0.033, 
                    num_leaves=8, 
                    colsample_bytree=0.2, 
                    subsample=0.01, 
                    max_depth=8, 
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
        sub_preds -= clf.predict_proba(test_data[feats].fillna(0.0).values, num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
        
        fold_importance = pd.DataFrame()
        fold_importance["feature"] = feats
        fold_importance["importance"] = clf.feature_importances_
        fold_importance["fold"] = n_fold + 1
        
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, train_x, train_y, val_x, val_y
        gc.collect()
#clf = LGBMClassifier(
#                    boosting_type = 'gbdt', 
#                    n_estimators=1000, 
#                    learning_rate=0.033, 
#                    num_leaves=8, 
#                    colsample_bytree=0.2, 
#                    subsample=0.01, 
#                    max_depth=8, 
#                    reg_alpha=.1, 
#                    reg_lambda=.03, 
#                    min_split_gain=.01, 
#                    min_child_weight=16, 
#                    silent=-1, 
#                    verbose=-1,
#                    random_state=546789
#                    )
#            
#clf.fit(X,y)
#
## Compute inverted predictions (to sort by later)
#test["predictions"] = -clf.predict_proba(test_data.fillna(0.0).values)[:, 1]
# Peek only needed columns and sort
test["predictions"] = sub_preds 
result = test[["instanceId_userId", "instanceId_objectId", "predictions"]].sort_values(
    by=['instanceId_userId', 'predictions'])
result.head(10)
# Collect predictions for each user
submit = result.groupby("instanceId_userId")['instanceId_objectId'].apply(list)
submit.head(10)
# Persist the first submit
submit.to_csv(output_path + "/collabSubmit.csv.gz", header = False, compression='gzip')        
