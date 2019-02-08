# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 14:26:42 2019

@author: gev
"""
# Used to read the Parquet data
import pyarrow.parquet as parquet
# Used to train the baseline model
from sklearn.linear_model import LogisticRegression

# Where the downloaded data are
input_path = 'e:/Other/Projects/MLBC/SNAHackathon2019/'
# Where to store results
output_path = 'e:/Other/Projects/MLBC/SNAHackathon2019/'

data = parquet.read_table(input_path + '/collabTrain/date=2018-02-07').to_pandas()
data.head(10)
data_10 = data.head(10)
# Construct the label (liked objects)
y = data['feedback'].apply(lambda x: 1.0 if("Liked" in x) else 0.0).values

# Extract the most interesting features
X = data[[
        'auditweights_svd_prelaunch', 
        'auditweights_ctr_high', 
        'auditweights_ctr_gender', 
        'auditweights_friendLikes']].fillna(0.0).values
# Fit the model and check the weights
model = LogisticRegression(random_state=0, solver='lbfgs').fit(X, y)
model.coef_
# Read the test data
test = parquet.read_table(input_path + '/collabTest').to_pandas()
test.head(10)

# Compute inverted predictions (to sort by later)
test["predictions"] = -model.predict_proba(test[[
        'auditweights_svd_prelaunch', 
        'auditweights_ctr_high', 
        'auditweights_ctr_gender', 
        'auditweights_friendLikes']].fillna(0.0).values)[:, 1]
# Peek only needed columns and sort
result = test[["instanceId_userId", "instanceId_objectId", "predictions"]].sort_values(
    by=['instanceId_userId', 'predictions'])
result.head(10)
# Collect predictions for each user
submit = result.groupby("instanceId_userId")['instanceId_objectId'].apply(list)
submit.head(10)
# Persist the first submit
submit.to_csv(output_path + "/collabSubmit.csv.gz", header = False, compression='gzip')        
