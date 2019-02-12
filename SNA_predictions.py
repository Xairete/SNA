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

data = parquet.read_table(input_path + '/collabTrain/date=2018-02-07').to_pandas()
data1 = parquet.read_table(input_path + '/collabTrain/date=2018-03-21').to_pandas()
data2 = parquet.read_table(input_path + '/collabTrain/date=2018-03-20').to_pandas()
data3 = parquet.read_table(input_path + '/collabTrain/date=2018-03-19').to_pandas()
data4 = parquet.read_table(input_path + '/collabTrain/date=2018-03-18').to_pandas()
data5 = parquet.read_table(input_path + '/collabTrain/date=2018-03-17').to_pandas()

data = pd.concat([data, data5])

del [data5]
data.head(20)
data.info()
data_10 = data.head(20)
# Construct the label (liked objects)
y = data['feedback'].apply(lambda x: 1.0 if("Liked" in x) else 0.0).values

# Extract the most interesting features
X = data[[
        'auditweights_svd_prelaunch', 
        'auditweights_ctr_high', 
        'auditweights_ctr_gender', 
        'auditweights_friendLikes',
        'auditweights_userOwner_TEXT',
        'auditweights_userOwner_CREATE_COMMENT',
        'auditweights_userOwner_CREATE_LIKE',
        'auditweights_userAge',
        'auditweights_userOwner_PHOTO_VIEW',
        'auditweights_userOwner_UNKNOWN',
        'auditweights_userOwner_USER_INTERNAL_LIKE',
        'auditweights_userOwner_USER_INTERNAL_UNLIKE',
        'auditweights_userOwner_VIDEO',
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

data.columns.values.tolist()

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
test["predictions"] = -clf.predict_proba(test[[
                'auditweights_svd_prelaunch', 
        'auditweights_ctr_high', 
        'auditweights_ctr_gender', 
        'auditweights_friendLikes',
        'auditweights_userOwner_TEXT',
        'auditweights_userOwner_CREATE_COMMENT',
        'auditweights_userOwner_CREATE_LIKE',
        'auditweights_userAge',
        'auditweights_userOwner_PHOTO_VIEW',
        'auditweights_userOwner_UNKNOWN',
        'auditweights_userOwner_USER_INTERNAL_LIKE',
        'auditweights_userOwner_USER_INTERNAL_UNLIKE',
        'auditweights_userOwner_VIDEO',
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
         'userOwnerCounters_VIDEO']].fillna(0.0).values)[:, 1]
# Peek only needed columns and sort
result = test[["instanceId_userId", "instanceId_objectId", "predictions"]].sort_values(
    by=['instanceId_userId', 'predictions'])
result.head(10)
# Collect predictions for each user
submit = result.groupby("instanceId_userId")['instanceId_objectId'].apply(list)
submit.head(10)
# Persist the first submit
submit.to_csv(output_path + "/collabSubmit.csv.gz", header = False, compression='gzip')        
