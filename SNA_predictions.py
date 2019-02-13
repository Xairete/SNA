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

del [data1, data2, data3, data4, data5, data6, data7]
data.info(max_cols=170)
data_10 = data.head(20)
# Construct the label (liked objects)
y = data['feedback'].apply(lambda x: 1.0 if("Liked" in x) else 0.0).values

data.select_dtypes(include=[object]).apply(pd.Series.nunique, axis = 0)

missing = missing_values_table(data)
missing_columns = list(missing[missing['% of Total Values'] > 99].index)
print('We will remove %d columns.' % len(missing_columns))
data = data.drop(columns = list(missing_columns))
data = data.drop(columns = 'metadata_options')
data = pd.get_dummies(data)

ids = data[['instanceId_userId', 'instanceId_objectId', 'audit_timestamp', 'audit_timePassed']]
data = data.drop(columns = ['instanceId_userId', 'instanceId_objectId', 'audit_timestamp', 'audit_timePassed'])

X = data.fillna(0.0).values

# Extract the most interesting features
X = data[[
        'auditweights_partAge',
        'auditweights_partCtr',
        'auditweights_partSvd',
         'auditweights_relationMasks',
         'auditweights_source_LIVE_TOP',
         'auditweights_source_MOVIE_TOP',

        'auditweights_svd_prelaunch', 
        'auditweights_ctr_high', 
        'auditweights_ctr_gender', 
        'auditweights_friendLikes',
        'auditweights_userOwner_TEXT',
        'auditweights_userOwner_CREATE_COMMENT',
        'auditweights_userOwner_CREATE_LIKE',
         'auditweights_userOwner_MOVIE_COMMENT_CREATE',
         'auditweights_userOwner_PHOTO_COMMENT_CREATE',
         'auditweights_userOwner_PHOTO_MARK_CREATE',
        
         'auditweights_numDislikes',
         'auditweights_numLikes',
         'auditweights_numShows',

        'auditweights_userAge',
        'auditweights_userOwner_PHOTO_VIEW',
        'auditweights_userOwner_UNKNOWN',
        'auditweights_userOwner_USER_INTERNAL_LIKE',
        'auditweights_userOwner_USER_INTERNAL_UNLIKE',
        'auditweights_userOwner_USER_PRESENT_SEND',
        'auditweights_userOwner_USER_PROFILE_VIEW',

        'auditweights_userOwner_VIDEO',
        'auditweights_userOwner_USER_FEED_REMOVE',
        'auditweights_x_ActorsRelations',
        'auditweights_friendLikes',
        'auditweights_numLikes',
        
        'userOwnerCounters_CREATE_LIKE',
        'userOwnerCounters_UNKNOWN',
        'userOwnerCounters_PHOTO_COMMENT_CREATE',
        'userOwnerCounters_USER_PHOTO_ALBUM_COMMENT_CREATE',
        'userOwnerCounters_USER_INTERNAL_LIKE',
        'userOwnerCounters_USER_INTERNAL_UNLIKE',
        'userOwnerCounters_MOVIE_COMMENT_CREATE',
        'userOwnerCounters_PHOTO_MARK_CREATE',
         'userOwnerCounters_CREATE_TOPIC',
         'userOwnerCounters_CREATE_IMAGE',
         'userOwnerCounters_CREATE_MOVIE',
         'userOwnerCounters_CREATE_COMMENT',
         'userOwnerCounters_TEXT',
         'userOwnerCounters_IMAGE',
         'userOwnerCounters_VIDEO'
         
        ]].fillna(0.0).values
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


data.columns.values.tolist()
data.info()

from lightgbm import LGBMClassifier

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
            
clf.fit(X,y)

# Compute inverted predictions (to sort by later)
test["predictions"] = -clf.predict_proba(test_data.fillna(0.0).values)[:, 1]
# Peek only needed columns and sort
result = test[["instanceId_userId", "instanceId_objectId", "predictions"]].sort_values(
    by=['instanceId_userId', 'predictions'])
result.head(10)
# Collect predictions for each user
submit = result.groupby("instanceId_userId")['instanceId_objectId'].apply(list)
submit.head(10)
# Persist the first submit
submit.to_csv(output_path + "/collabSubmit.csv.gz", header = False, compression='gzip')        
