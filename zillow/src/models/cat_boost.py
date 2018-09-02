import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
import re
from ggplot import *
import datetime
from category_encoders import BinaryEncoder
from sklearn.preprocessing import LabelBinarizer,Imputer,OneHotEncoder,LabelEncoder
#from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin



print( "\nReading data from disk ...")
train=pd.read_csv('C:/Users/Lenovo/Downloads/all/train_2016_v2.csv',parse_dates=['transactiondate'])
properties=pd.read_csv('C:/Users/Lenovo/Downloads/all/properties_2016.csv')
sample=pd.read_csv('C:/Users/Lenovo/Downloads/all/sample_submission.csv')

for c, dtype in zip(properties.columns, properties.dtypes):
    if dtype == np.float64:
        properties[c] = properties[c].astype(np.float32)

#droping redundant columns

drop_cols=['calculatedbathnbr','finishedsquarefeet50','fips','censustractandblock']

#updating 'pooltypeid10'=1 when 'pooltypeid2'=1
properties['pooltypeid10']=np.where(properties['pooltypeid2']==1,1,0)

#updating 'poolcnt'=1 when 'pooltypeid2'=1
properties['poolcnt']=np.where(properties['pooltypeid2']==1,1,0)

#updating 'poolcnt'=1 when 'pooltypeid7'=1
properties['poolcnt']=np.where(properties['pooltypeid7']==1,1,0)

#updating NaN values to 0
properties['pooltypeid7']=np.where(properties['pooltypeid7'].isnull(),0,1)

#updating NaN values to 0
properties['pooltypeid2']=np.where(properties['pooltypeid2'].isnull(),0,1)

#If there is no pool pool size must be zero
properties['poolsizesum']=np.where(properties['poolcnt'].isnull(),0,1)

#updating airconditioningtypeid to 6, which is 'other' category, refering zillow data dictionary
index = properties.airconditioningtypeid.isnull()
properties.loc[index,'airconditioningtypeid'] = 6

#The below variables are flags and lets assume if they are NA's it means the object does not exist so lets fix this
#index = properties.hashottuborspa.isnull()
#properties.loc[index,'hashottuborspa'] = "None"

#updating airconditioningtypeid to 6, which is 'other' category, refering zillow data dictionary
index = properties.heatingorsystemtypeid.isnull()
properties.loc[index,'heatingorsystemtypeid'] = 14


#updating taxdelinquencyflag to valid flag based on the value of taxdelinquencyyear
properties['taxdelinquencyflag']=np.where(properties['taxdelinquencyyear'].isnull(),'None','True')

#updating taxdelinquencyflag to valid flag based on the value of taxdelinquencyyear
index = properties.fireplacecnt.isnull()
properties.loc[index,'fireplacecnt'] = 0

properties['fireplaceflag']=np.where(properties['fireplacecnt']==0,'None','True')

#if 'bathroomcnt'=0 then 'fullbathcnt' and 'threequarterbathnbr' should be zero too

index = properties.bathroomcnt.isnull()
properties.loc[index,'fullbathcnt'] = 0
properties.loc[index,'threequarterbathnbr'] = 0

index = properties.roomcnt.isnull()
properties.loc[index,'roomcnt']=properties['roomcnt'].median()

#Feature Engineering

properties['Age_of_Home']=2018-properties['yearbuilt']

properties.drop(columns=drop_cols,inplace=True)

train_merge = train.merge(properties, how='left', on='parcelid')
#ulimit = np.percentile(train_merge.logerror.values, 99)
#llimit = np.percentile(train_merge.logerror.values, 1)
#train_merge['logerror'].ix[train_merge['logerror']>ulimit] = ulimit
#train_merge['logerror'].ix[train_merge['logerror']<llimit] = llimit
mean = np.mean(train_merge['logerror'], axis=0)
sd = np.std(train_merge['logerror'], axis=0)
llimit=mean - 3 * sd
ulimit=mean + 3 * sd
#train_merge['logerror'].ix[train_merge['logerror']>ulimit] = ulimit
#train_merge['logerror'].ix[train_merge['logerror']<llimit] = llimit

# drop out ouliers
train_merge=train_merge[ train_merge.logerror > llimit]
train_merge=train_merge[ train_merge.logerror < ulimit ]
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


exclude_other = ['parcelid', 'logerror','transactiondate','regionidcounty']
train_features = []
for c in train_merge.columns:
    if c not in exclude_missing \
       and c not in exclude_other \
        and c not in exclude_unique :
        train_features.append(c)
print("We use these for training: %s" % train_features)
print(len(train_features))


#Time to train the model

X_train = train_merge[train_features]
y_train = train_merge.logerror
print(X_train.shape, y_train.shape)
X_train.info()


sample['parcelid'] = sample['ParcelId']
test_merge = sample.merge(properties, on='parcelid', how='left')
#for c in cat_feature_inds:
#    dummy=pd.get_dummies(test_merge[c],prefix=c)
#    test_merge=pd.concat([test_merge,dummy],axis=1)
#test_merge.drop(['hashottuborspa', 'fireplaceflag', 'taxdelinquencyflag'],axis=1,inplace=True)

x_test = test_merge[train_features]
y_train=train_merge.logerror


X_train.fillna(-999,inplace=True)
x_test.fillna(-999,inplace=True)


num_ensembles = 5
y_pred = 0.0
cat_feature_inds=[21,23,34,37]
for i in range(num_ensembles):
    model = CatBoostRegressor(
        iterations=630, learning_rate=0.03,
        depth=6, l2_leaf_reg=3,
        loss_function='MAE',
        eval_metric='MAE',
        random_seed=i)
    model.fit(
        X_train, y_train,cat_feature_inds)
    y_pred += model.predict(X_test)
y_pred /= num_ensembles

submission = pd.DataFrame({
    'ParcelId': test_merge['ParcelId'],
})
test_dates = {
    '201610': pd.Timestamp('2016-09-30'),
    '201611': pd.Timestamp('2016-10-31'),
    '201612': pd.Timestamp('2016-11-30'),
    '201710': pd.Timestamp('2017-09-30'),
    '201711': pd.Timestamp('2017-10-31'),
    '201712': pd.Timestamp('2017-11-30')
}
for label, test_date in test_dates.items():
    print("Predicting for: %s ... " % (label))
    submission[label] = y_pred

submission.to_csv('Only_CatBoost.csv', float_format='%.6f',index=False)
