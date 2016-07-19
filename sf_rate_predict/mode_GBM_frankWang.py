
# coding: utf-8

# # sf work assignment.6.8.2016: Part II: model
# Frank Wang

# In[1]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import random
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error


# # Funtions

# In[2]:

import sklearn.metrics
import time
def evaluate_model(clf):
    """Scores a model using log loss with the created train and test sets."""
    start = time.time()
    clf.fit(x_train, y_train)
    print ("Train score:", sklearn.metrics.roc_auc_score(y_train, clf.predict(x_train)))
    print ("Test score:", sklearn.metrics.roc_auc_score(y_test, clf.predict(x_test)))
    print ("Total time:", time.time() - start)


# In[3]:

def shuffle(df):
    index = list(df.index)
    random.shuffle(index)
    df = df.ix[index]
    df.reset_index()
    return df


# In[4]:

def r_squared(y_pred,y_true):
    ymean=np.sum(y_true)/len(y_true)
    sstot=np.sum((y_true-ymean)**2)
    ssreg=np.sum((y_true-y_pred)**2) 
    result=1.0-ssreg/sstot
    return result


# In[5]:

def rmse(y_pred,y_true):
    rmse= mean_squared_error(y_true, y_pred)**0.5
    return rmse


# # train data

# In[6]:

train_data=pd.read_csv('featured_Train.csv',sep=',')    
test_data=pd.read_csv('featured_Test.csv',sep=',')    
print len(train_data)
print len(test_data)
test_data['category_educational']=0


# In[7]:

shuffle(train_data)
print 'done'


# In[8]:

target=train_data.rate[train_data.dateLoan<4]
train_data_new=train_data.drop('rate',axis=1)[train_data.dateLoan<4]
print len(train_data_new)


# In[9]:

import sklearn
import sklearn.cross_validation
from sklearn.linear_model import LogisticRegression

np.random.seed(1333)
x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(train_data_new,target,test_size=1.0/3.0,                                                                             random_state=12)


# ### linear model

# In[41]:

from sklearn import linear_model
ols = linear_model.LinearRegression()
ols.fit(train_data_new,target)
y_train_out = ols.predict(x_train)
y_out = ols.predict(x_test)
#evaluate_model(ols)
print "R^2 for train set:",
print ols.score(x_train,y_train)
print "R^2 for test set:",
print ols.score(x_test,y_test)
# R^2 for train set: 0.988827381996
# R^2 for test set: 0.988746258488
#print r_squared(y_out,y_test)
print "RMSE of train",
print rmse(y_train_out,y_train)
print "RMSE of test",
print rmse(y_out,y_test)


# In[51]:

test_all=1.0*train_data_new
y_linear_all=ols.predict(train_data_new)
test_all['rmse']=y_train_out = y_linear_all-target
StateGroup=test_all.groupby(['dateLoan']).mean()
my_xticks = StateGroup.index
x= np.arange(len(my_xticks))
x2=x+0.5
fig=plt.figure()
plt.bar(StateGroup['rmse'].keys(),StateGroup['rmse'].values,0.1)
plt.gca().invert_xaxis()
#plt.xticks(x, my_xticks,fontsize = 15,rotation='vertical')
#plt.xticks(x2, my_xticks,fontsize = 8,rotation='horizontal')
plt.yticks(fontsize = 15)
plt.xticks(fontsize = 15)
plt.xlabel('dateLoan (year from now)',fontsize=25)
plt.ylabel('mean error (%)',fontsize=25)
fig.set_size_inches(12,6) 
plt.title('Mean error',fontsize=20)
dump=['dateLoan','rmse']
temp=test_all[dump]
temp.to_csv('test_rmse.csv',sep=' ')


# In[59]:

fig=plt.figure()
plt.plot(y_linear_all-target,'k.',markersize=1)
plt.ylim([-2,2])
fig.set_size_inches(12,6) 
plt.xlabel('Sample ID',fontsize=20)
plt.ylabel('Residual',fontsize=20)
plt.title('Linear model',fontsize=20)
plt.yticks(fontsize = 15)
plt.xticks(fontsize = 15)


# In[12]:

elastic = linear_model.ElasticNet(l1_ratio =0.5,normalize=True)
elastic.set_params(alpha = 1.0e-7)
elastic.fit(x_train,y_train)
y_pred_test_ela = elastic.predict(x_test)
y_pred_train_ela = elastic.predict(x_train)
#evaluate_model(ols)
print "R^2 for train set:",
print elastic.score(x_train,y_train)
print "R^2 for test set:",
print elastic.score(x_test,y_test)
#print r_squared(y_out,y_test)
print "RMSE of train",
print rmse(y_pred_train_ela,y_train)
print "RMSE of test",
print rmse(y_pred_test_ela,y_test)


# ## Gradient Booster regression

# In[13]:

from sklearn.ensemble import GradientBoostingRegressor
import time
gbr = GradientBoostingRegressor()


# In[14]:

#gbr_best = GradientBoostingRegressor(n_estimators=300, learning_rate=0.2,max_depth=3)
start = time.time()
gbr_best=GradientBoostingRegressor(alpha=0.9, init=None, learning_rate=0.15, loss='ls',
             max_depth=6, max_features=None, max_leaf_nodes=None,
             min_samples_leaf=1, min_samples_split=4,
             min_weight_fraction_leaf=0.0, n_estimators=300,
             presort='auto', random_state=None, subsample=0.9, verbose=0,
             warm_start=False)
gbr_best.fit(train_data_new,target)
pred_G = gbr_best.predict(x_test)
print gbr_best.score(x_test,y_test)
print gbr_best.score(x_train,y_train)
print "Total time:", time.time() - start


# In[16]:

y_tr_gbr=gbr_best.predict(x_train)
y_te_gbr=gbr_best.predict(x_test)
y_all_gbr=gbr_best.predict(train_data_new)
print "RMSE of train",
print rmse(y_tr_gbr,y_train)
print "RMSE of test",
print rmse(y_te_gbr,y_test)
print "RMSE of whole data",
print rmse(y_all_gbr,target)


# In[17]:

from sklearn.externals import joblib
joblib.dump(gbr_best, 'gbm_best_final.pkl') 
y_test_gbr_out=gbr_best.predict(test_data)
with open('FrankWang_GBM.csv','a') as f:
    for (i,z) in enumerate(y_test_gbr_out):
        f.write('{}%'.format(z))
        f.write('\n')
f.close()


# In[58]:

fig=plt.figure()
plt.plot(y_all_gbr-target,'k.',markersize=1)
plt.ylim([-2,2])
plt.xlabel('Sample ID',fontsize=20)
plt.ylabel('Residual',fontsize=20)
plt.title('GBM',fontsize=20)
plt.yticks(fontsize = 15)
plt.xticks(fontsize = 15)
fig.set_size_inches(12,6) 


# In[54]:

test_all=1.0*train_data_new
test_all['rmse']=y_all_gbr-target
StateGroup=test_all.groupby(['dateLoan']).mean()
my_xticks = StateGroup.index
x= np.arange(len(my_xticks))
x2=x+0.5
fig=plt.figure()
plt.bar(StateGroup['rmse'].keys(),StateGroup['rmse'].values,0.1)
plt.gca().invert_xaxis()
#plt.xticks(x, my_xticks,fontsize = 15,rotation='vertical')
#plt.xticks(x2, my_xticks,fontsize = 8,rotation='horizontal')
plt.yticks(fontsize = 15)
plt.xticks(fontsize = 15)
plt.xlabel('dateLoan (year from now)',fontsize=25)
plt.ylabel('mean error (%)',fontsize=25)
fig.set_size_inches(12,6) 
plt.title('Mean error distribution with time',fontsize=20)
dump=['dateLoan','rmse']
temp=test_all[dump]
temp.to_csv('train_gbm_rmse.csv',sep=' ')


# In[20]:

feature_importance = gbr_best.feature_importances_
num=len(gbr_best.feature_importances_)
num_plot=15
# make importances relative to max importance
sorted_idx = np.argsort(feature_importance)[0:num]  ##increase order
pos = np.arange(sorted_idx.shape[0]) + .5

fig = plt.figure(figsize=(10,30))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, x_train.columns[[sorted_idx]])
plt.xlabel('Relative Importance',fontsize=20)
plt.yticks(fontsize=20)
plt.title('GBM Variable Importance',fontsize=20)
plt.show()

fig.set_dpi(1240)
fig.savefig('feature_importance_GBR_morefeature.png', transparent=True, bbox_inches='tight', pad_inches=0)


# ###### gridsearch

# In[ ]:

max_depth_values = [3,4,5,6,7]
learning_rate_values = [0.15,0.2,0.25]
subsample_values = [0.8,0.9,1]
min_samples_split_values =[2,4]
params = {'max_depth' : max_depth_values, 'learning_rate': learning_rate_values, 
          'subsample': subsample_values, 'min_samples_split': min_samples_split_values}
grid = GridSearchCV(GradientBoostingRegressor(n_estimators=300), params)
grid.fit(x_train,y_train)
scores_test=grid.score(x_test,y_test)
scores_train=grid.score(x_train,y_train)
print ('done')


# In[ ]:

print grid.grid_scores_
print grid.score(x_test,y_test)


# ## XGBOOST

# In[21]:

import xgboost as xgb
import sklearn.cross_validation as cv
import scipy as sp


# In[22]:

param = {'objective': 'reg:linear',
              'eta': 0.12,
              'eval_metric':'rmse',
              'subsample': 0.7,
              'max_depth': 7,
              'min_child_weight': 2,
              'colsample_bytree': 0.5,
              'gamma': 5,
              'silent': 1
              }
folds = 5 
num_round = 280


# In[23]:

xgmat = xgb.DMatrix(train_data_new, label=target)
watchlist =[(xgmat, 'train')]

bst_best_model= xgb.train(param, xgmat, num_round, watchlist)
print('model fit done')
#     save out model
bst_best_model.save_model('best_xgb.model')
xgmat_test = xgb.DMatrix(x_test)
pred = bst_best_model.predict(xgmat)


# In[24]:

y_xgb=bst_best_model.predict(xgmat)
print "RMSE of train",
print rmse(y_xgb,target)
print "R2 of train",
print r_squared(y_xgb,target)


# In[30]:

feature_train=train_data_new.columns
test_data=test_data[feature_train]


# In[31]:

from sklearn.externals import joblib
joblib.dump(bst_best_model, 'xgb_best_model.pkl') 
xgmat_holdout = xgb.DMatrix(test_data)
y_test_xgb_out=bst_best_model.predict(xgmat_holdout)
with open('FrankWang_XGB.csv','a') as f:
    for (i,z) in enumerate(y_test_xgb_out):
        f.write('{}%'.format(z))
        f.write('\n')
f.close()


# In[56]:

fig=plt.figure()
plt.plot(y_xgb-target,'k.',markersize=1)
plt.ylim([-2,2])
fig.set_size_inches(12,6)
plt.xlabel('Sample ID',fontsize=20)
plt.ylabel('Residual',fontsize=20)
plt.title('XGB',fontsize=20)
plt.yticks(fontsize = 15)
plt.xticks(fontsize = 15)


# In[57]:

test_all=1.0*train_data_new
test_all['rmse']=y_all_gbr-target
StateGroup=test_all.groupby(['dateLoan']).mean()
my_xticks = StateGroup.index
x= np.arange(len(my_xticks))
x2=x+0.5
fig=plt.figure()
plt.bar(StateGroup['rmse'].keys(),StateGroup['rmse'].values,0.1)
plt.gca().invert_xaxis()
#plt.xticks(x, my_xticks,fontsize = 15,rotation='vertical')
#plt.xticks(x2, my_xticks,fontsize = 8,rotation='horizontal')
plt.yticks(fontsize = 15)
plt.xticks(fontsize = 15)
plt.xlabel('dateLoan (year from now)',fontsize=25)
plt.ylabel('mean error (%)',fontsize=25)
fig.set_size_inches(12,6) 
plt.title('Mean error distribution with time',fontsize=20)
dump=['dateLoan','rmse']
temp=test_all[dump]
temp.to_csv('test_xgb_rmse.csv',sep=' ')


# ###### grid search

# In[107]:

param = {'objective': 'reg:linear',
              'eta': 0.12,
              'eval_metric':'rmse',
              'subsample': 0.7,
              'max_depth': 7,
              'min_child_weight': 2,
              'colsample_bytree': 0.5,
              'gamma': 5,
              'silent': 1
              }
folds = 5 
num_round = 180


# In[ ]:

f=open("summary_bst_scan.txt","a")
f.write('tuning parameters=')
f.write('\n')
for key, values in param.items():
    f.write('param {}={}'.format(key,values))
    f.write('\n')
f.write('nfolds=%f' %(folds))
f.write('\n')
f.write('num_round=%f' %(num_round))
f.write('\n')
f.close()
max_depth_values=[7,8]
min_child_weight_values=[2,3,4]
max_depth_values=[7]
min_child_weight_values=[2]

#     for x in max_depth_values:
#         for y in min_child_weight_values:
#             f=open("summary_bst_scan.txt","a")
#             f.write('\n')
#             f.write('-------------')
#             f.write('\n')
#             param['max_depth']=x
#             param['min_child_weight']=y
#             f.write('param {}={}'.format('max_depth',x))
#             f.write('\n')
#             f.write('param {}={}'.format('mini_child_weight',y))
#             f.write('\n')
#             f.write('--------------')
#             f.write('\n')
#             f.close()
#     gamma_vlaues=[10,100]
#     for x in gamma_vlaues:
#             f=open("summary_bst_scan.txt","a")
#             f.write('\n')
#             f.write('-------------')
#             f.write('\n')
#             param['gamma']=x
#             f.write('param {}={}'.format('gamma',x))
#             f.write('\n')
#             f.write('--------------')
#             f.write('\n')
#             f.close()
#     subsample_values=[0.6,0.8,0.9,1]
#     colsample_bytree_values=[0.6,0.8,0.9,1]
# #     subsample_values=[1]
# #     colsample_bytree_values=[0.5,0.7]
#     for x in subsample_values:
#         for y in colsample_bytree_values:
#             f=open("summary_bst_scan.txt","a")
#             f.write('\n')
#             f.write('-------------')
#             f.write('\n')
#             param['subsample']=x
#             param['colsample_bytree']=y
#             f.write('param {}={}'.format('subsample',x))
#             f.write('\n')
#             f.write('param {}={}'.format('colsample_bytree',y))
#             f.write('\n')
#             f.write('--------------')
#             f.write('\n')
#             f.close()
#     num_round_values =[150,180,200,250]
#     num_round_values =[100,120,130,140]
num_round_values =[260,300,340]
for x in num_round_values:
        f=open("summary_bst_scan.txt","a")
        f.write('\n')
        f.write('-------------')
        f.write('\n')
        num_round =x
        f.write('num_round={}'.format(x))
        f.write('\n')
        f.write('--------------')
        f.write('\n')
        f.close()

        estimate_performance_xgboost(train_data_new, target.values, param, num_round, folds)


# In[109]:

def estimate_performance_xgboost(X,labels,param, num_round, folds):
    '''
    Cross validation for XGBoost performance
    '''
    f=open("summary_bst_scan.txt","a")
    start = np.random.random_integers(1000) #time.time()
    # Cross validate
    kf = cv.KFold(labels.size, n_folds=folds, random_state=start)
    # Dictionary to store all the AMSs
    all_rmse = []
    for train_indices, test_indices in kf:
        X_train, X_test = X.loc[train_indices], X.loc[test_indices]
        y_train, y_test = labels[train_indices], labels[test_indices]
        xgmat = xgb.DMatrix(X_train, label=y_train)
        plst = param.items()#+[('eval_metric', 'ams@0.15')]

        watchlist = []#[(xgmat, 'train')]
        bst = xgb.train(plst, xgmat, num_round, watchlist)

        xgmat_test = xgb.DMatrix(X_test)
        y_out = bst.predict(xgmat_test)
        num=y_test.shape[0]
        y_test=np.reshape(y_test,num)
        rmse_score=rmse(y_out,y_test)
        print('rmse={}'.format(rmse_score))
        f.write('rmse={}'.format(rmse_score))
        f.write('\n')
        all_rmse.append(rmse_score)
    print ("------------------------------------------------------")
    print ("mean rmse ={} with std={}".format(sp.mean(all_rmse),sp.std(all_rmse)))
    f.write("mean rmse ={} with std={}".format(sp.mean(all_rmse),sp.std(all_rmse)))
    f.write('\n')   
    f.close()


# In[ ]:



