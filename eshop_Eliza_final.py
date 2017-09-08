#!/usr/bin/python



#Import models from scikit learn module:
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample 
from sklearn import linear_model
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import tree,ensemble
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from datetime import timedelta
import time
pd.options.mode.chained_assignment=None

#Generic function for making a classification model and accessing performance:
def classification_model(model, data_s, data, data_test, predictors, outcome):
#Fit the model:

  x=data[predictors].fillna(value=np.mean(data[predictors])) # train data
  x_t=data_test[predictor_var].fillna(value=np.mean(data_test[predictor_var])) #test data
  x_s=data_s[predictors].fillna(value=np.mean(data_s[predictors])) # x downsampled of train data to build model
  #x_s_train=x_s[:int(x_s.shape[0]*0.66)]
  #x_s_test=x[:int(x.shape[0]*0.33)]  
  #print x
  y_s=data_s[outcome] # y downsampled of train data to build model

  y=data[outcome] # y train data to calculate the accuracy or evluate the model
  #y_s_train=y_s[:int(y_s.shape[0]*0.66)]
  #y_s_test=y[:int(y.shape[0]*0.33)]  
  model.fit(x_s,y_s) #model is based on downsampled data
  
  
  predictions = model.predict(x) #Make predictions on training set not on downsampled data:



  predictions_test = model.predict(x_t)  #predict results for test.csv

    
  
  accuracy = metrics.accuracy_score(predictions,y) #Print accuracy on the whole training data
  
  print ("Accuracy : %s" % "{0:.3%}".format(accuracy))
    
  # check AUROC (area under ROC Curve)
  prob_y_2 = model.predict_proba(x)
  prob_y_2 = [p[1] for p in prob_y_2]
  print ("AUROC : ",roc_auc_score(y, prob_y_2) )
    
  #Perform k-fold cross-validation with 10 folds
  kf = KFold(data.shape[0], n_folds=10)
  print (kf)
  error = []
  for train, test in kf:
    # Filter training data
    #print 'train:', train
    #print 'test:', test
    train_predictors = (x.iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = y[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(x.iloc[test,:], y[test]))
    
 
  print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

    
  #Fit the model again so that it can be refered outside the function:
  model.fit(x,y)  

  return predictions_test # return the prediction for test data 
    
#----------------------------------------------------------------------------------
# importinig all data set
df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')
df_transaction=pd.read_csv('transactions.csv')
df_userfile=pd.read_csv('user_profiles.csv')

now = datetime.now()
ttoday=int(time.time())

#---------------------------------------------------------------------------------------------
# the part mainly create features for each user based on transaction history
# groupd the information together to list for same userid
df_history=df_transaction.groupby('userid', as_index=False).agg(lambda x: x.tolist())

#df_ is generated from df_history but without lists.
columnsh = ['userid','spent_trend','order_number', 'promotion_used_number','last_order','last_order_day','day_counts','shop_diversity','repeat_shop']
indexh=df_history.index
df_ = pd.DataFrame(index=indexh, columns=columnsh) # create a new dataframe df_
df_ = df_.fillna(0) # with 0s rather than NaNs to initialize
df_['userid']=df_history['userid']

#integrate some information from df_history to df_
for index, row in df_history.iterrows():
    t=np.count_nonzero(~np.isnan(row['promotionid_used']))
    df_['promotion_used_number'][index] = t # the number of promotion each user used
    df_['order_number'][index]=len(row['promotionid_used']) # total number of orders for each user
    df_['last_order_day'][index]=(ttoday-max(row['order_time']))/86400 # last order until now unit days 
    df_['last_order'][index]=max(row['order_time'])       # last order exact time
    df_['day_counts'][index]=(max(row['order_time'])-min(row['order_time']))/86400    # day counts of each user  
    df_['shop_diversity'][index]=len(np.unique(row['shopid']))     # shop diversity per user
    df_['repeat_shop'][index]= row['shopid'].count(max(row['shopid'],key=row['shopid'].count))   # max frequency to some shop per user
    time = row['order_time']
    s = row['total_price']
    time, s = zip(*sorted(zip(time, s)))
    tm=min(time, key=lambda x:abs(x-round((max(time)-min(time))/2+min(time))))
    ind=time.index(tm)
    #delt_s=np.mean(s[ind:])-np.mean(s[:ind])
    if (max(row['order_time'])-min(row['order_time']))/86400 > 30 and len(s[ind:]) != 0 and len(s[:ind]) != 0:
        df_.loc[index,'spent_trend']=round(np.mean(s[ind:])-np.mean(s[:ind]))  # latter averge spent - early average spent by each user
df_['spent_trend']=df_['spent_trend'].fillna(-10) # not well distributed orders
#------------------------------------------------------------
# I tried to convert consecutive values to bins in this part, but it takes time to formulate based on distribution or logically
# Since I didnt formulate them beautifully, I didnt use df_1 later
# convert the consective numbers in total spent, order number and promotion used number to catogeries using bins
# here I generated 10 bins, this parameter can be changed later
bins = np.zeros((3,21))
df_1=pd.DataFrame(index=indexh, columns=columnsh)
df_1 = df_1.fillna(0) # with 0s rather than NaNs
df_1['userid']=df_history['userid']
df_1['last_order']=df_['last_order']
for a, k in zip(df_[df_.columns[1:4]], range(len(df_.columns)-1)):
    #print np.linspace(min(df_[a]), max(df_[a]), 11)
    #print a
    bins[k]=np.linspace(min(df_[a]), max(df_[a]), 21)
    #print bins[k]
    which_bin = np.digitize(df_[a], bins=bins[k])
    #print which_bin
    df_1[a]=which_bin
    
#----------------------------------------------------------------
#this part df_userfile adapt the register date and age
# only year will be consider here


df_userfile['registration_y'] = pd.to_datetime(df_userfile['registration_time'])
df_userfile['registration_y']= df_userfile['registration_y'].map(lambda x: 1*x.year )
df_userfile['registration_y']=now.year-df_userfile['registration_y']   #the number of years that registered

df_userfile['age'] = pd.to_datetime(df_userfile['birthday'])
df_userfile['age']= df_userfile['age'].map(lambda x: 1*x.year )
df_userfile['age'].replace([-1],[1901])
df_userfile['age']=now.year-df_userfile['age']
df_userfile['age']=df_userfile['age'].replace([2018,116],[None,None])
df_userfile['age'][df_userfile['age']< 0 ] = 0
df_userfile['age'][df_userfile['age']> 80] = 0
df_userfile['age']=df_userfile['age'].fillna(0)
bins=np.linspace(min(df_userfile['age']), max(df_userfile['age']), 5)
print bins
which_bin = np.digitize(df_userfile['age'], bins=bins)
df_userfile['age']=which_bin



df_userfile.drop(['phone_verified','registration_time','birthday',],inplace=True,axis=1) # delete some feature that not be used later

df_userfile['gender'].replace([3,4],[1,2],inplace=True) # replace the predicted gender to real gender
df_userfile['gender']=df_userfile['gender'].fillna(value=0) # replace missing value to 0

#-----------------------------------------------------------------------------------------------
# this part merge files together based on user id
# merge userprofile to train df_train
df_T=df_train.merge(df_userfile, left_on='userid', right_on='userid', how='left')
# merge transaction file to train df_train
df_T=df_T.merge(df_, left_on='userid', right_on='userid', how='left')

df_T['waiting_time']=((df_T['voucher_received_time'].astype(float)-df_T['last_order'].astype(float))/86400)
#----------------------------------------------------------
# merge userprofile and transaction file to test data df_test
df_Test=df_test.merge(df_userfile, left_on='userid', right_on='userid', how='left')
# merge transaction file to train df_train
df_Test=df_Test.merge(df_, left_on='userid', right_on='userid', how='left')

df_Test['waiting_time']=((df_Test['voucher_received_time'].astype(float)-df_Test['last_order'].astype(float))/86400)

#----------------------------------------------------------------------------------
# here I try to encode values 
#var_mod = ['promotionid_received','email_verified','is_seller','total_spent','order_number', 'promotion_used_number']
var_mod = ['email_verified','is_seller']
le = LabelEncoder()

for i in var_mod:
    df_T[i] = le.fit_transform(df_T[i])
    df_Test[i] = le.fit_transform(df_Test[i])
#print (df_T['promotion_used_number'].unique().shape)
#---------------------------------------------------------------------------
# imbarassing data will be processed in this part
#down sample the data to handle imbalanced classes or  downSAMPLE THE UNBALANCED DATA

# Separate majority and minority classes for train data used
df_majority_T_u = df_T[df_T['used?']==0]
df_minority_T_u = df_T[df_T['used?']==1]


# Separate majority and minority classes for train data repurchase
df_majority_T_r = df_T[df_T['repurchase?']==1]
df_minority_T_r = df_T[df_T['repurchase?']==0]

# Downsample minority class
df_majority_T_downsampled_u = resample(df_majority_T_u, 
                                 replace=False,     # sample with replacement
                                 n_samples=2738*2,    # to match majority class
                                 random_state=10) # reproducible results

# Downsample minority class
df_majority_T_downsampled_r = resample(df_majority_T_r, 
                                 replace=False,     # sample with replacement
                                 n_samples=13552*1.2,    # to match majority class
                                 random_state=10) # reproducible results

# Combine majority class with downsampled minority class
df_downsampled_T_u = pd.concat([df_minority_T_u, df_majority_T_downsampled_u])
df_downsampled_T_u.index = range(df_downsampled_T_u.shape[0]) #reset index to 0,1,2... 
 
# Combine majority class with downsampled minority class
df_downsampled_T_r = pd.concat([df_minority_T_r, df_majority_T_downsampled_r])
df_downsampled_T_r.index = range(df_downsampled_T_r.shape[0]) #reset index to 0,1,2... 
# Display new class counts
print df_downsampled_T_r['repurchase?'].value_counts()
print df_downsampled_T_u['used?'].value_counts()
#df_downsampled_T.to_csv('downsampled.csv') 


#---------------------------------------------------------------------------

model_0 = linear_model.LogisticRegression(n_jobs=4)
#model = tree.DecisionTreeClassifier( min_samples_split=25, max_depth=7)
#model = ensemble.RandomForestClassifier(n_estimators=100) # accuracy 100% but always overfitting
model = ensemble.RandomForestClassifier(n_estimators=20, min_samples_split=25, criterion='entropy',min_samples_leaf=5,max_depth=7, max_features=4,n_jobs=4)
#model = AdaBoostClassifier(n_estimators=100,learning_rate=1, random_state=0)
model_1 = GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=1, random_state=0)


#model_e = VotingClassifier(estimators=[('lr', model_0),('rf', model), ('gb', model_1)], voting='soft', weights=[1,1,1])
model_e = VotingClassifier(estimators=[('rf', model), ('gb', model_1)], voting='soft', weights=[1,1])


#params = {'lr__C': [1.0, 100.0], 'rf__n_estimators': [10, 200],}
#grid = GridSearchCV(estimator=model_e, param_grid=params, cv=5)




#print (model)


#predictor_var = ['total_spent','order_number','promotion_used_number']
predictor_var = ['email_verified','registration_y','age','is_seller', 'gender','spent_trend','order_number',
                 'promotion_used_number','waiting_time','shop_diversity','repeat_shop','last_order_day','day_counts']


print (predictor_var)
outcome_var_used = 'used?'
outcome_var_pur='repurchase?'






predictions_repurchase=classification_model(model_e, df_downsampled_T_r,df_T,df_Test,predictor_var,outcome_var_pur)
#Create a series with feature importances:
#featimp = pd.Series(model.feature_importances_, index=predictor_var)
#print ("Summary :\n",featimp)
#------------------------------------------------------------------------------------------------------------
predictions_used=classification_model(model_e, df_downsampled_T_u,df_T,df_Test,predictor_var,outcome_var_used)

#featimp = pd.Series(model_1.feature_importances_, index=predictor_var)
#print ("Summary :\n",featimp)

df_Test['repurchase?']=predictions_repurchase
df_Test['used?']=predictions_used
df_Test.to_csv('test_prediction_reslut.csv') 



