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

import gc
gc.enable()

all_list = ['instanceId_userId', 'instanceId_objectType', 'instanceId_objectId', 'audit_pos', 'audit_clientType', 'audit_timestamp', 'audit_timePassed',
         'audit_experiment', 'audit_resourceType', 'metadata_ownerId', 'metadata_ownerType', 'metadata_createdAt', 'metadata_authorId', 'metadata_applicationId',
         'metadata_numCompanions', 'metadata_numPhotos', 'metadata_numPolls', 'metadata_numSymbols', 'metadata_numTokens', 'metadata_numVideos', 'metadata_platform',
         'metadata_totalVideoLength', 'metadata_options', 'relationsMask', 'userOwnerCounters_USER_FEED_REMOVE', 'userOwnerCounters_USER_PROFILE_VIEW',
         'userOwnerCounters_VOTE_POLL', 'userOwnerCounters_USER_SEND_MESSAGE', 'userOwnerCounters_USER_DELETE_MESSAGE', 'userOwnerCounters_USER_INTERNAL_LIKE',
         'userOwnerCounters_USER_INTERNAL_UNLIKE', 'userOwnerCounters_USER_STATUS_COMMENT_CREATE', 'userOwnerCounters_PHOTO_COMMENT_CREATE', 'userOwnerCounters_MOVIE_COMMENT_CREATE',
         'userOwnerCounters_USER_PHOTO_ALBUM_COMMENT_CREATE', 'userOwnerCounters_COMMENT_INTERNAL_LIKE', 'userOwnerCounters_USER_FORUM_MESSAGE_CREATE',
         'userOwnerCounters_PHOTO_MARK_CREATE', 'userOwnerCounters_PHOTO_VIEW', 'userOwnerCounters_PHOTO_PIN_BATCH_CREATE', 'userOwnerCounters_PHOTO_PIN_UPDATE',
         'userOwnerCounters_USER_PRESENT_SEND', 'userOwnerCounters_UNKNOWN', 'userOwnerCounters_CREATE_TOPIC', 'userOwnerCounters_CREATE_IMAGE',
         'userOwnerCounters_CREATE_MOVIE', 'userOwnerCounters_CREATE_COMMENT', 'userOwnerCounters_CREATE_LIKE', 'userOwnerCounters_TEXT', 'userOwnerCounters_IMAGE',
         'userOwnerCounters_VIDEO', 'ownerUserCounters_USER_FEED_REMOVE', 'ownerUserCounters_USER_PROFILE_VIEW', 'ownerUserCounters_VOTE_POLL', 'ownerUserCounters_USER_SEND_MESSAGE',
         'ownerUserCounters_USER_DELETE_MESSAGE', 'ownerUserCounters_USER_INTERNAL_LIKE', 'ownerUserCounters_USER_INTERNAL_UNLIKE', 'ownerUserCounters_USER_STATUS_COMMENT_CREATE',
         'ownerUserCounters_PHOTO_COMMENT_CREATE', 'ownerUserCounters_MOVIE_COMMENT_CREATE', 'ownerUserCounters_USER_PHOTO_ALBUM_COMMENT_CREATE',
         'ownerUserCounters_COMMENT_INTERNAL_LIKE', 'ownerUserCounters_USER_FORUM_MESSAGE_CREATE', 'ownerUserCounters_PHOTO_MARK_CREATE', 'ownerUserCounters_PHOTO_VIEW',
         'ownerUserCounters_PHOTO_PIN_BATCH_CREATE', 'ownerUserCounters_PHOTO_PIN_UPDATE', 'ownerUserCounters_USER_PRESENT_SEND', 'ownerUserCounters_UNKNOWN',
         'ownerUserCounters_CREATE_TOPIC', 'ownerUserCounters_CREATE_IMAGE', 'ownerUserCounters_CREATE_MOVIE', 'ownerUserCounters_CREATE_COMMENT',
         'ownerUserCounters_CREATE_LIKE', 'ownerUserCounters_TEXT', 'ownerUserCounters_IMAGE', 'ownerUserCounters_VIDEO', 'membership_status', 'membership_statusUpdateDate',
         'membership_joinDate', 'membership_joinRequestDate', 'owner_create_date', 'owner_birth_date', 'owner_gender', 'owner_status', 'owner_ID_country',
         'owner_ID_Location', 'owner_is_active', 'owner_is_deleted', 'owner_is_abused', 'owner_is_activated', 'owner_change_datime', 'owner_is_semiactivated',
         'owner_region', 'user_create_date', 'user_birth_date', 'user_gender', 'user_status', 'user_ID_country', 'user_ID_Location', 'user_is_active',
         'user_is_deleted', 'user_is_abused', 'user_is_activated', 'user_change_datime', 'user_is_semiactivated', 'user_region', 'feedback', 'objectId',
         'auditweights_ageMs', 'auditweights_closed', 'auditweights_ctr_gender', 'auditweights_ctr_high', 'auditweights_ctr_negative', 'auditweights_dailyRecency',
         'auditweights_feedOwner_RECOMMENDED_GROUP', 'auditweights_feedStats', 'auditweights_friendCommentFeeds', 'auditweights_friendCommenters',
         'auditweights_friendLikes', 'auditweights_friendLikes_actors', 'auditweights_hasDetectedText', 'auditweights_hasText', 'auditweights_isPymk',
         'auditweights_isRandom', 'auditweights_likersFeedStats_hyper', 'auditweights_likersSvd_prelaunch_hyper', 'auditweights_matrix',
         'auditweights_notOriginalPhoto', 'auditweights_numDislikes', 'auditweights_numLikes', 'auditweights_numShows', 'auditweights_onlineVideo',
         'auditweights_partAge', 'auditweights_partCtr', 'auditweights_partSvd', 'auditweights_processedVideo', 'auditweights_relationMasks',
         'auditweights_source_LIVE_TOP', 'auditweights_source_MOVIE_TOP', 'auditweights_svd_prelaunch', 'auditweights_svd_spark', 'auditweights_userAge',
         'auditweights_userOwner_CREATE_COMMENT', 'auditweights_userOwner_CREATE_IMAGE', 'auditweights_userOwner_CREATE_LIKE', 'auditweights_userOwner_IMAGE',
         'auditweights_userOwner_MOVIE_COMMENT_CREATE', 'auditweights_userOwner_PHOTO_COMMENT_CREATE', 'auditweights_userOwner_PHOTO_MARK_CREATE',
         'auditweights_userOwner_PHOTO_VIEW', 'auditweights_userOwner_TEXT', 'auditweights_userOwner_UNKNOWN', 'auditweights_userOwner_USER_DELETE_MESSAGE',
         'auditweights_userOwner_USER_FEED_REMOVE', 'auditweights_userOwner_USER_FORUM_MESSAGE_CREATE', 'auditweights_userOwner_USER_INTERNAL_LIKE',
         'auditweights_userOwner_USER_INTERNAL_UNLIKE', 'auditweights_userOwner_USER_PRESENT_SEND', 'auditweights_userOwner_USER_PROFILE_VIEW',
         'auditweights_userOwner_USER_SEND_MESSAGE', 'auditweights_userOwner_USER_STATUS_COMMENT_CREATE', 'auditweights_userOwner_VIDEO', 'auditweights_userOwner_VOTE_POLL',
         'auditweights_x_ActorsRelations', 'auditweights_likersSvd_spark_hyper', 'auditweights_source_PROMO']

missing_columns = ['relationsMask', 'ownerUserCounters_PHOTO_PIN_UPDATE','owner_is_activated',
         'owner_is_abused','owner_is_deleted', 'owner_is_active', 'owner_ID_Location',  'owner_ID_country',  'owner_status',
         'owner_gender', 'owner_birth_date', 'owner_create_date', 'ownerUserCounters_VIDEO', 'ownerUserCounters_IMAGE', 'ownerUserCounters_TEXT',
         'ownerUserCounters_CREATE_LIKE', 'ownerUserCounters_CREATE_COMMENT', 'ownerUserCounters_CREATE_MOVIE', 'ownerUserCounters_CREATE_IMAGE', 'ownerUserCounters_CREATE_TOPIC',
         'ownerUserCounters_UNKNOWN', 'owner_change_datime', 'owner_is_semiactivated', 'auditweights_closed', 'auditweights_userOwner_USER_DELETE_MESSAGE',
         'auditweights_userOwner_VOTE_POLL', 'auditweights_userOwner_USER_STATUS_COMMENT_CREATE', 'auditweights_userOwner_USER_SEND_MESSAGE',
         'auditweights_userOwner_USER_PROFILE_VIEW', 'auditweights_userOwner_USER_PRESENT_SEND', 'auditweights_userOwner_USER_INTERNAL_UNLIKE',
         'auditweights_userOwner_USER_INTERNAL_LIKE', 'auditweights_userOwner_USER_FORUM_MESSAGE_CREATE', 'auditweights_userOwner_PHOTO_VIEW',
         'auditweights_isPymk', 'auditweights_userOwner_PHOTO_MARK_CREATE', 'auditweights_userOwner_PHOTO_COMMENT_CREATE',
         'auditweights_userOwner_MOVIE_COMMENT_CREATE', 'auditweights_source_LIVE_TOP', 'auditweights_relationMasks',
         'auditweights_partSvd', 'auditweights_partCtr', 'auditweights_partAge', 'ownerUserCounters_USER_PRESENT_SEND', 'owner_region',
         'ownerUserCounters_PHOTO_PIN_BATCH_CREATE', 'ownerUserCounters_USER_INTERNAL_UNLIKE', 'ownerUserCounters_PHOTO_VIEW',
         'ownerUserCounters_USER_FEED_REMOVE', 'ownerUserCounters_USER_PROFILE_VIEW', 'ownerUserCounters_VOTE_POLL',
         'ownerUserCounters_USER_SEND_MESSAGE', 'ownerUserCounters_USER_DELETE_MESSAGE', 'ownerUserCounters_USER_INTERNAL_LIKE',
         'auditweights_source_PROMO', 'ownerUserCounters_PHOTO_MARK_CREATE', 'ownerUserCounters_USER_STATUS_COMMENT_CREATE',
         'ownerUserCounters_PHOTO_COMMENT_CREATE', 'ownerUserCounters_MOVIE_COMMENT_CREATE', 'ownerUserCounters_USER_PHOTO_ALBUM_COMMENT_CREATE',
         'ownerUserCounters_COMMENT_INTERNAL_LIKE', 'ownerUserCounters_USER_FORUM_MESSAGE_CREATE', 'auditweights_hasDetectedText',
         'auditweights_source_MOVIE_TOP', 'auditweights_userOwner_CREATE_IMAGE', 'auditweights_onlineVideo', 'auditweights_userOwner_VIDEO',
         'auditweights_friendCommentFeeds', 'auditweights_friendCommenters', 'auditweights_userOwner_UNKNOWN']


select_list = list(set(all_list) - set(missing_columns)) 

from datetime import date, timedelta, time
oldest = date(2018,3,21)
data_sample = parquet.read_table(input_path + '/collabTrain/date=2018-03-21', columns = select_list).to_pandas()
   
dayofweek = oldest.weekday()
data_sample['dayofweek'] = str(dayofweek)
data_sample['day'] = oldest
for i in range(1,21):
    print(oldest - timedelta(i))
    day  = oldest - timedelta(i)
    dayofweek = day.weekday()
    s = '/collabTrain/date='+str((oldest - timedelta(i)))
    data1 = parquet.read_table(input_path + s, columns = select_list).to_pandas()
    data1['dayofweek'] = str(dayofweek)
    data1['day'] = day
    data_sample = pd.concat([data_sample, data1])
    del data1

feed = data_sample['feedback']

y = feed.apply(lambda x: 1.0 if("Liked" in x) else 0.0)
data_sample['liked'] = y.rename('liked').astype('Int16')
data_sample = data_sample.drop(columns = 'feedback')

data = data_sample.sample(frac = 0.20, random_state=546789)
data = data_sample
valid_data = data
y_all = data['liked']

data.info(max_cols = 172)
data.day = pd.to_datetime(data.day)
isweekend = pd.to_datetime(data.day).dt.dayofweek.apply(lambda x: 1.0 if(x==6 or x==5) else 0.0)
data['isweekend'] = isweekend

#________________________________________________________________

User_Object_count = data[['instanceId_userId','instanceId_objectId']].groupby('instanceId_userId').count().astype('Int16')
Object_User_count = data[['instanceId_userId','instanceId_objectId']].groupby('instanceId_objectId').count().astype('Int16')
User_Object_count = User_Object_count.rename(columns = {'instanceId_objectId':'User_Object_count'})
Object_User_count = Object_User_count.rename(columns = {'instanceId_userId':'Object_User_counter'})
data = data.join(User_Object_count, on = 'instanceId_userId')
data = data.join(Object_User_count, on = 'instanceId_objectId')

data = data.drop(columns =['liked'])

data.info(max_cols=170)
data_20 = data.head(50)

missing = missing_values_table(data)
missing_columns = list(missing[missing['% of Total Values'] > 99].index)
print('We will remove %d columns.' % len(missing_columns))

option = data.metadata_options.apply(lambda x: 1.0 if("HAS_TEXT" in x) else 0.0)
data['HAS_TEXT'] = option
sum_of_option = option
option = data.metadata_options.apply(lambda x: 1.0 if("HAS_PHOTOS" in x) else 0.0)
data['HAS_PHOTOS'] = option
sum_of_option += option
option = data.metadata_options.apply(lambda x: 1.0 if("HAS_POLLS" in x) else 0.0)
data['HAS_POLLS'] = option
sum_of_option += option
option = data.metadata_options.apply(lambda x: 1.0 if("HAS_VIDEOS" in x) else 0.0)
data['HAS_VIDEOS'] = option
sum_of_option += option
option = data.metadata_options.apply(lambda x: 1.0 if("HAS_URLS" in x) else 0.0)
data['HAS_URLS'] = option
sum_of_option += option
option = data.metadata_options.apply(lambda x: 1.0 if("IS_PART_OF_ALBUM" in x) else 0.0)
data['IS_PART_OF_ALBUM'] = option
sum_of_option += option
option = data.metadata_options.apply(lambda x: 1.0 if("IS_PART_OF_TOPIC" in x) else 0.0)
data['IS_PART_OF_TOPIC'] = option
sum_of_option += option
option = data.metadata_options.apply(lambda x: 1.0 if("IS_EXTERNAL_SHARE" in x) else 0.0)
data['IS_EXTERNAL_SHARE'] = option
sum_of_option += option
option = data.metadata_options.apply(lambda x: 1.0 if("IS_INTERNAL_SHARE" in x) else 0.0)
data['IS_INTERNAL_SHARE'] = option
sum_of_option += option
option = data.metadata_options.apply(lambda x: 1.0 if("HAS_DETECTED_TEXT" in x) else 0.0)
data['HAS_DETECTED_TEXT'] = option
sum_of_option += option
data['sum_of_options'] = sum_of_option 

data = data.drop(columns = list(missing_columns))
ids = data[['audit_experiment','metadata_options','instanceId_userId', 'instanceId_objectId', 'audit_timestamp', 'audit_timePassed']]
data = data.drop(columns = ['audit_experiment','metadata_options','instanceId_userId', 'instanceId_objectId', 'audit_timestamp', 'audit_timePassed'])
#data.metadata_createdAt = pd.to_datetime(data.metadata_createdAt, unit='ms')
data = pd.get_dummies(data)
#_____________________________________________________________________________
# Fit the model and check the weight
# Read the test data
test = parquet.read_table(input_path + '/collabTest', columns = list(set(select_list)|set(['date']))).to_pandas()
test.head(10)

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

option = test.metadata_options.apply(lambda x: 1.0 if("HAS_TEXT" in x) else 0.0)
test['HAS_TEXT'] = option
sum_of_option = option
option = test.metadata_options.apply(lambda x: 1.0 if("HAS_PHOTOS" in x) else 0.0)
test['HAS_PHOTOS'] = option
sum_of_option += option
option = test.metadata_options.apply(lambda x: 1.0 if("HAS_POLLS" in x) else 0.0)
test['HAS_POLLS'] = option
sum_of_option += option
option = test.metadata_options.apply(lambda x: 1.0 if("HAS_VIDEOS" in x) else 0.0)
test['HAS_VIDEOS'] = option
sum_of_option += option
option = test.metadata_options.apply(lambda x: 1.0 if("HAS_URLS" in x) else 0.0)
test['HAS_URLS'] = option
sum_of_option += option
option = test.metadata_options.apply(lambda x: 1.0 if("IS_PART_OF_ALBUM" in x) else 0.0)
test['IS_PART_OF_ALBUM'] = option
sum_of_option += option
option = test.metadata_options.apply(lambda x: 1.0 if("IS_PART_OF_TOPIC" in x) else 0.0)
test['IS_PART_OF_TOPIC'] = option
sum_of_option += option
option = test.metadata_options.apply(lambda x: 1.0 if("IS_EXTERNAL_SHARE" in x) else 0.0)
test['IS_EXTERNAL_SHARE'] = option
sum_of_option += option
option = test.metadata_options.apply(lambda x: 1.0 if("IS_INTERNAL_SHARE" in x) else 0.0)
test['IS_INTERNAL_SHARE'] = option
sum_of_option += option
option = test.metadata_options.apply(lambda x: 1.0 if("HAS_DETECTED_TEXT" in x) else 0.0)
test['HAS_DETECTED_TEXT'] = option
sum_of_option += option
test['sum_of_options'] = sum_of_option 

test = test.drop(columns = 'date')
#test.metadata_createdAt = pd.to_datetime(test.metadata_createdAt, unit='ms')
test_data = test #.drop(columns = list(missing_columns))

ids_list = ['audit_experiment','metadata_options','instanceId_userId', 'instanceId_objectId', 'audit_timestamp', 'audit_timePassed']
test_data = test_data.drop(columns = ['audit_experiment','metadata_options','instanceId_userId', 'instanceId_objectId', 'audit_timestamp', 'audit_timePassed'])

test_data = pd.get_dummies(test_data)

print('Training Features shape: ', data.shape)
print('Testing Features shape: ', test_data.shape)
data.info(max_cols=210)
test_data.info(max_cols=210)
data['label'] = y_all
corr_koef = data.corr()
field_drop = [i for i in corr_koef if corr_koef[i].isnull().drop_duplicates().values[0]]
cor_field = []
for i in corr_koef:
    for j in corr_koef.index[abs(corr_koef[i]) > 0.9]:
        if i != j and j not in cor_field and i not in cor_field:
            cor_field.append(j)
            print ("%s-->%s: r^2=%f" % (i,j, corr_koef[i][corr_koef.index==j].values[0]))
            
field_drop =field_drop + cor_field
field_drop = ['auditweights_isRandom', 'userOwnerCounters_USER_PRESENT_SEND', 'auditweights_hasText',
             'userOwnerCounters_USER_STATUS_COMMENT_CREATE', 'auditweights_notOriginalPhoto', 'userOwnerCounters_PHOTO_PIN_BATCH_CREATE',
             'userOwnerCounters_VOTE_POLL', 'userOwnerCounters_MOVIE_COMMENT_CREATE', 'userOwnerCounters_PHOTO_PIN_UPDATE',
             'userOwnerCounters_COMMENT_INTERNAL_LIKE', 'userOwnerCounters_USER_INTERNAL_LIKE', 'userOwnerCounters_USER_DELETE_MESSAGE',
             'userOwnerCounters_USER_PHOTO_ALBUM_COMMENT_CREATE', 'userOwnerCounters_PHOTO_COMMENT_CREATE', 'userOwnerCounters_USER_SEND_MESSAGE',
             'userOwnerCounters_USER_PROFILE_VIEW', 'metadata_applicationId', 'auditweights_processedVideo',
             'userOwnerCounters_PHOTO_VIEW', 'userOwnerCounters_USER_INTERNAL_UNLIKE', 'userOwnerCounters_USER_FORUM_MESSAGE_CREATE',
             'userOwnerCounters_PHOTO_MARK_CREATE', 'membership_joinRequestDate', 'membership_statusUpdateDate', 'metadata_numTokens',
             'user_birth_date', 'HAS_POLLS', 'instanceId_objectType_Post', 'metadata_ownerType_GROUP_OPEN_OFFICIAL']
train_list = data.columns.values.tolist() 
test_list = test_data.columns.values.tolist() 
for j in test_list:
    if j not in train_list:
        print(j)
data = data.drop(field_drop, axis=1)
test_data = test_data.drop(field_drop, axis=1)
data = data.drop(columns = 'membership_status_R')

corr = data.corr().ix['label', :-1]
import matplotlib.pyplot as plt
plt.hist(data['auditweights_svd_prelaunch'].fillna(-1), 20)

data['label'] = y_all
data['instanceId_userId'] = ids['instanceId_userId']
data['instanceId_objectId'] = ids['instanceId_objectId']


Xfilt = data.loc[~(data['userOwnerCounters_CREATE_LIKE'] > 8000)]
Xfilt =Xfilt.loc[~(Xfilt['auditweights_ctr_high'] < 0)]


#metadata_numSymbols
#Xfilt['userOwnerCounters_CREATE_COMMENT'] = np.log(Xfilt['userOwnerCounters_CREATE_COMMENT']+1)
#
#Xfilt = data.loc[~(data['userOwnerCounters_VIDEO'] > 1000)]
#Xfilt =Xfilt.loc[~(Xfilt['userOwnerCounters_IMAGE'] > 21000)]
#Xfilt =Xfilt.loc[~(Xfilt['userOwnerCounters_TEXT'] > 12500)]
#Xfilt =Xfilt.loc[~(Xfilt['userOwnerCounters_CREATE_COMMENT'] > 10000)]
#Xfilt =Xfilt.loc[~(Xfilt['userOwnerCounters_CREATE_TOPIC'] > 2000)]
#Xfilt =Xfilt.loc[~(Xfilt['auditweights_ctr_high'] < 0)]
#Xfilt =Xfilt.loc[~(Xfilt['auditweights_likersSvd_spark_hyper']>3)]
#Xfilt =Xfilt.loc[~(Xfilt['metadata_numPhotos'] > 150)]
#Xfilt =Xfilt.loc[~(Xfilt['metadata_totalVideoLength'] > 60000000)]
#Xfilt =Xfilt.loc[~(Xfilt['userOwnerCounters_USER_FEED_REMOVE'] > 15000)]
#Xfilt =Xfilt.loc[~(Xfilt['userOwnerCounters_CREATE_LIKE'] > 8000)]

import seaborn as sns
boxplot = data.hist(column=[ 'auditweights_likersSvd_spark_hyper'])

#Xfilt = data
X = Xfilt.fillna(0.0)
T = test_data.fillna(0.0)

#X = data
#T = test_data
valid_data = X
y = X['label']
X = X.drop(columns = 'label')

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
support_feats = ['audit_pos','audit_resourceType', 'metadata_ownerId', 'metadata_authorId' , 'metadata_createdAt',
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
                 'metadata_platform_OTHER', 'metadata_platform_WEB', 'membership_status_A' #]
                 ,'HAS_TEXT', 'HAS_PHOTOS', 'HAS_VIDEOS', 'HAS_URLS','IS_PART_OF_ALBUM', 'IS_PART_OF_TOPIC', 'IS_EXTERNAL_SHARE','IS_INTERNAL_SHARE','HAS_DETECTED_TEXT', 'sum_of_options']
                 

X = X.drop(columns = ['day', 'metadata_createdAt'])

import time
T = T.drop(columns = ['day', 'metadata_createdAt'])
#feats = data.columns.values.tolist() 
feats = [f for f in X.columns if f in support_feats]
oof_preds = np.zeros(X.shape[0])
sub_preds = np.zeros(T.shape[0])
sub_valid = 0 

from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=546789)
for n_fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
        train_x, train_y = X[feats].iloc[train_idx], y.iloc[train_idx]
        val_x, val_y = X[feats].iloc[val_idx], y.iloc[val_idx]

        clf = LGBMClassifier(
                    boosting_type = 'gbdt', 
                    n_estimators=2000, 
                    learning_rate=0.1, 
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
        sub_valid_data = valid_data[["instanceId_userId", "instanceId_objectId", 'label']].iloc[val_idx]
        sub_valid_data['score'] = oof_preds[val_idx]        
        
        sub_valid += sub_valid_data.groupby("instanceId_userId")\
            .apply(lambda y: auc(y.label.values, y.score.values))\
            .dropna().mean()
            
        #print('Sub AUC : %.6f' % sub_valid)
        sub_preds -= clf.predict_proba(T[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance = pd.DataFrame()
        fold_importance["feature"] = feats
        fold_importance["importance"] = clf.feature_importances_
        fold_importance["fold"] = n_fold + 1
        
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del  train_x, train_y, val_x, val_y, clf
        gc.collect()

print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))  
print('Full AUC score %.6f' % (sub_valid/5))  

valid_data['score'] = oof_preds

valid = valid_data.groupby("instanceId_userId")\
    .apply(lambda y: auc(y.label.values, y.score.values))\
    .dropna().mean()

test["predictions"] = sub_preds 

result = test[["instanceId_userId", "instanceId_objectId", "predictions"]].sort_values(
    by=['instanceId_userId', "predictions"])
submit = result.groupby("instanceId_userId")['instanceId_objectId'].apply(list)
submit.to_csv(output_path + "/before_replace_collabSubmit.csv.gz", header = False, compression='gzip')  
#_______________________________________________________________________________
oldest = date(2018,3,21)
data_sample = parquet.read_table(input_path + '/collabTrain/date=2018-03-21', columns = ["instanceId_userId", "instanceId_objectId", 'feedback']).to_pandas()
dayofweek = oldest.weekday()
data_sample['dayofweek'] = str(dayofweek)
data_sample['day'] = oldest
for i in range(1,48):
    print(oldest - timedelta(i))
    day  = oldest - timedelta(i)
    dayofweek = day.weekday()
    if (str((oldest - timedelta(i))) != '2018-02-11'):
        s = '/collabTrain/date='+str((oldest - timedelta(i)))
        data1 = parquet.read_table(input_path + s, columns = ["instanceId_userId", "instanceId_objectId", 'feedback']).to_pandas()
        data1['dayofweek'] = str(dayofweek)
        data1['day'] = day
        data_sample = pd.concat([data_sample, data1])

feed = data_sample['feedback']
del data1

y = feed.apply(lambda x: 1.0 if("Liked" in x) else 0.0)
data_sample['liked'] = y.rename('liked').astype('Int16')

#data = data_sample.sample(frac = 0.20, random_state=546789)
data = data_sample


concatdata = data["instanceId_userId"].apply(str) +'_'+ data["instanceId_objectId"].apply(str)
concattest = test["instanceId_userId"].apply(str) +'_'+ test["instanceId_objectId"].apply(str)
d = pd.Series(list(set(concatdata) & set(concattest)))
data['concatdata'] = concatdata
test['concattest'] = concattest
data_concat = (data[['concatdata', 'liked']].groupby('concatdata').median())
test1 = test[['concattest',"predictions"]]
w = test1.join(data_concat, how='left', on='concattest')
w.liked.fillna(w.predictions, inplace=True)

test['liked'] = w['liked'].apply(lambda x: -1.0 if (x == 1.0) else x )

#result = test[["instanceId_userId", "instanceId_objectId", "predictions"]].sort_values(
#    by=['instanceId_userId', 'predictions'])

result = test[["instanceId_userId", "instanceId_objectId", "liked"]].sort_values(
    by=['instanceId_userId', 'liked'])

result.head(10)    
# Collect predictions for each user

submit = result.groupby("instanceId_userId")['instanceId_objectId'].apply(list)
submit.head(10)
# Persist the first submit
submit.to_csv(output_path + "/after_replace_collabSubmit.csv.gz", header = False, compression='gzip')   

from sklearn.linear_model import Ridge
ridgereg = Ridge(alpha=0.1,normalize=True)

del data_sample
del data
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
