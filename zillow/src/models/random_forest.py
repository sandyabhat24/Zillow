import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
#from category_encoders import BinaryEncoder
#from sklearn.preprocessing import LabelBinarizer,Imputer,OneHotEncoder,LabelEncoder
#from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
#from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import zillow.src.config.config as config
#import config.transform as impute
#from config.config import config
import zillow.src.config.transform as impute




print( "\nReading data from disk ...")
train=pd.read_csv(config.DATA_PATHS['train_2016_v2'],parse_dates=['transactiondate'])
properties=pd.read_csv(config.DATA_PATHS['properties'])
sample=pd.read_csv(config.DATA_PATHS['sample'])


for c, dtype in zip(properties.columns, properties.dtypes):
    if dtype == np.float64:
        properties[c] = properties[c].astype(np.float32)

impute.transform()
train_merge = train.merge(properties, how='left', on='parcelid')
impute.outlier()
#We will drop any coulmn with more than 99% missing values.

missing_perc_thresh = 0.99
exclude_missing = []
num_rows = train_merge.shape[0]
for c in train_merge.columns:
    num_missing = train_merge[c].isnull().sum()
    if num_missing == 0:
        continue
    missing_frac = num_missing / float(num_rows)
    if missing_frac > missing_perc_thresh:
        exclude_missing.append(c)
print("We exclude: %s" % exclude_missing)
print(len(exclude_missing))

# exclude where we only have one unique value
exclude_unique = []
for c in train_merge.columns:
    num_uniques = len(train_merge[c].unique())
    if train_merge[c].isnull().sum() != 0:
        num_uniques -= 1
    if num_uniques == 1:
        exclude_unique.append(c)
print("We exclude: %s" % exclude_unique)
print(len(exclude_unique))

#Selecting categorical variables for model building.
exclude_other = ['transactiondate','propertycountylandusecode','propertyzoningdesc']
cat_feature_inds = []
col=train_merge.columns
cat_unique_thresh = 1000
for c in train_merge.columns:
     if train_merge[c].dtype == 'object' \
            and c not in exclude_other:
        cat_feature_inds.append(c)
print("Cat features are: %s" %  cat_feature_inds)

#Handling categorical variables

for c in cat_feature_inds:
    dummy=pd.get_dummies(train_merge[c],prefix=c)
    train_merge=pd.concat([train_merge,dummy],axis=1)
train_merge.drop(['hashottuborspa', 'fireplaceflag', 'taxdelinquencyflag'],axis=1,inplace=True)

exclude_other = ['parcelid', 'logerror','transactiondate','regionidcounty']
train_features = []
for c in train_merge.columns:
    if c not in exclude_missing \
       and c not in exclude_other \
        and c not in exclude_unique and train_merge[c].dtype != 'object':
        train_features.append(c)
print("We use these for training: %s" % train_features)
print(len(train_features))


#Time to train the model

X_train = train_merge[train_features]
y_train = train_merge.logerror
print(X_train.shape, y_train.shape)


sample['parcelid'] = sample['ParcelId']
test_merge = sample.merge(properties, on='parcelid', how='left')
for c in cat_feature_inds:
    dummy=pd.get_dummies(test_merge[c],prefix=c)
    test_merge=pd.concat([test_merge,dummy],axis=1)
test_merge.drop(['hashottuborspa', 'fireplaceflag', 'taxdelinquencyflag'],axis=1,inplace=True)

x_test = test_merge[train_features]
y_train=train_merge.logerror


X_train.fillna(-999,inplace=True)
x_test.fillna(-999,inplace=True)

pipe_rf = Pipeline([('scl', StandardScaler()),
                ('clf', RandomForestRegressor(random_state=42))])

# Set grid search params
#bstrap=[True, False]
grid_param_rf = {# Number of trees in random forest
                'clf__n_estimators': [int(x) for x in np.linspace(start = 100, stop = 300, num = 12)],
                # Number of features to consider at every split
                'clf__max_features': ['auto', 'sqrt'],
                # Maximum number of levels in tree
                'clf__max_depth' : [int(x) for x in np.linspace(5, 30, num = 6)],
                #max_depth.append(None)
                # Minimum number of samples required to split a node
                'clf__min_samples_split' : np.arange(2, 5, 10),
                # Minimum number of samples required at each leaf node
                'clf__min_samples_leaf' : np.arange(1, 2, 4)
                # Method of selecting samples for training each tree
                #'bootstrap' : bstrap
}
gs_rf = RandomizedSearchCV(estimator=pipe_rf,
             param_distributions=grid_param_rf,
             n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)

gs_rf.fit(X_train, y_train)
y_test = gs_rf.predict(x_test)
y_test = pd.DataFrame(y_test)
y_test[1] = y_test[0]
y_test[2] = y_test[0]
y_test[3] = y_test[0]
y_test[4] = y_test[0]
y_test[5] = y_test[0]
y_test.columns = ["201610","201611","201612","201710","201711","201712"]
submission = y_test.copy()
submission["parcelid"] = sample["ParcelId"].copy()
cols = ["parcelid","201610","201611","201612","201710","201711","201712"]
201610, 201611, 201612, 201701, 201702, 201703
submission = submission[cols]
filename = "Prediction_" + str(submission.columns[0]) + re.sub("[^0-9]", "",str(datetime.datetime.now())) + '.csv'
print(filename)
submission.to_csv(filename,index=False)
print( "\nFinished ..." )
