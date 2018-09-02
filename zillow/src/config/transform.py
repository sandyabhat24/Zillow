def transform():
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
    index = properties.hashottuborspa.isnull()
    properties.loc[index,'hashottuborspa'] = "None"

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



def outlier():
    mean = np.mean(train_merge['logerror'], axis=0)
    sd = np.std(train_merge['logerror'], axis=0)
    llimit=mean - 3 * sd
    ulimit=mean + 3 * sd
    train_merge['logerror'].ix[train_merge['logerror']>ulimit] = ulimit
    train_merge['logerror'].ix[train_merge['logerror']<llimit] = llimit
