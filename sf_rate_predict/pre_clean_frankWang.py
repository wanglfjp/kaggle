import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import random
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error


'''
data type convert:
convert percentage into faction; date to number of years; rate to number; dollar amount to float
'''
from re import sub
from decimal import Decimal
def dollarString2float(string):
    '''
    Remove dollar sign from the string and return the number.
    Return -999 if there is a missing value.
    @param: raw string
    @return: number
    '''
    string=str(string)
    if string[0]!='$':
        return -999
    else:
        return float(Decimal(sub(r'[^\d.]', '', string)))
def monthString2float(string):
    '''
    Remove month from the string and return the number.
    Return -999 if there is a missing value.
    @param: raw string
    @return: number
    '''
    string=str(string)
    tmp=re.findall(r'\d\d.', string)
    if tmp:
        return float(Decimal(sub(r'[^\d.]', '', string)))
    else:
        return -999
def rateString2float(string):
    '''
    Remove % sign from the string and return the number.
    @param: raw string
    @return: number
    '''
    string=str(string)
    return (float(string.strip('%')))
	
def employyear_String2float(string):
    '''
    Remove "<,years,+" from the string and return the number.
    Return -999 if there is a missing value.
    @param: raw string
    @return: number
    '''
    string=str(string)
    string=string.strip('years').strip().strip('<').strip('+').strip('\n').strip('n/')
    if len(string)>0:
        try:
            aa=float(string)
        except ValueError:
            aa=-999
        return aa
    else:
        return -999
		
from datetime import datetime, timedelta
def mon_year2months(datestring):
    '''
    convert date type "Jun-09" from the string and return the number.
    Return -999 if there is a missing value.
    @param: raw string
    @return: number of years from that date to today, 2016-07-07
    the function name "2month" is kind of missleading, since oroginally it return month in stead of year;
    we keep like this.
    '''
    datestring=str(datestring)
    tmp=re.findall('\w\w\w', datestring)
    if tmp:
        monthdate=tmp[0]
        if len(monthdate)!=3:
            return -999
    else:
        return -999    
    tmp=re.findall('\d+',datestring)
    if tmp:
        yeardate=tmp[0]
        if len(yeardate)!=2:
            return -999
    else:
        return -999
    if int(yeardate)<16:
        yeardate='20'+yeardate     
    else:
        yeardate='19'+yeardate
    monthd1=datetime.strptime(monthdate, '%b').month
    day1='1'
    string_date=yeardate+'-'+str(monthd1)+'-'+day1
    date_object = datetime.strptime(string_date, '%Y-%m-%d')
    datetime_now= datetime.strptime('2016-07-07', '%Y-%m-%d')
    dt=datetime_now-date_object
    return (dt.days/365.0)

def transform_data(df):
    '''
    Clean the input pandas dataframe using functions defined previously
    @param: pandas dataframe
    @return: pandas dataframe
    '''
    df.loanReq=df.loanReq.apply(dollarString2float)
    df.loanFund=df.loanFund.apply(dollarString2float)
    df.investFrac=df.investFrac.apply(dollarString2float)
    df.numPayment=df.numPayment.apply(monthString2float)
    df.dateCreditOpen=df.dateCreditOpen.apply(mon_year2months)
    df.dateLoan=df.dateLoan.apply(mon_year2months)
    df.rate=df.rate.apply(rateString2float)
    df.debt2limitRatio=df.debt2limitRatio.apply(rateString2float)*0.01
    df.yearsEmployed=df.yearsEmployed.apply(employyear_String2float)


	
def main():
	read_all=1
	num_data=200000
	if read_all==1:
		train_data=pd.read_csv('C:/wanglf2016/kaggle/state_fram_assignment/data/Data for Cleaning & Modeling.csv',\
		                      sep=',',header=0)    
	else:
		train_data=pd.read_csv('C:/wanglf2016/kaggle/state_fram_assignment/data/Data for Cleaning & Modeling.csv',sep=',',header=0,\
                            nrows=num_data)
	test_data=pd.read_csv('C:/wanglf2016/kaggle/state_fram_assignment/data/Holdout for Testing.csv',sep=',',header=0)
	num_train=len(train_data)
	num_test=len(test_data)

	features_names=['rate','loanID','borrowerID','loanReq','loanFund','investFrac', \
               'numPayment','grade','subGrade','employer','yearsEmployed','homeOwner',\
                'income','incomeVeri','dateLoan','reason','category','loanTitle',
                'zipcode','state','debtRatio','dueNum','dateCreditOpen','numInquiry',\
               'monthsDelinquency','monthsRecord','numOpenCredit','numDerogatory',\
               'creditBalance','debt2limitRatio','numTotalCredit','listStatus']
	train_data.columns=features_names
	test_data.columns=features_names
	transform_data(train_data)
	transform_data(test_data)
	train_data=train_data.replace(-999.000,np.nan)
	print 'data transform done'

	num_miss_col=np.sum(train_data.isnull())
	num_miss_row=np.sum(train_data.isnull().any(axis=1))

	feature_missing=[]
	for (it,feature) in enumerate(features_names):
		if (num_miss_col[it]>num_train*0.5):
			feature_missing.append(feature)
			print ('feature {} has more than 50% missing'.format(feature))
		
	if len(set(train_data.loanID))!=len(train_data.loanID):
		print ('some repeat ID found')
	if len(set(train_data.borrowerID))!=len(train_data.borrowerID):
		print ('some repeat borroweID found')
		
	train_data=train_data[train_data.rate.notnull()]
	
	for (it,feature) in enumerate(train_data.columns):
		if (num_miss_col[it]<num_train*0.05):
			train_data=train_data[train_data.iloc[:,it].notnull()]
			print ('drop rows: feature {} has {} missing'.format(feature,num_miss_col[it]))       
	train_data=train_data[train_data.subGrade.notnull()]
	# rare event, not in test
	state_drop=['IA','ID','ME','NE']
	train_data=train_data[train_data.state != state_drop[0]]
	train_data=train_data[train_data.state != state_drop[1]]
	train_data=train_data[train_data.state != state_drop[2]]
	train_data=train_data[train_data.state != state_drop[3]]
	owner_keep=['MORTGAGE','OWN','RENT']
	owner_drop=['ANY','NONE','OTHER'] 
	train_data=train_data[train_data.homeOwner != owner_drop[0]]
	train_data=train_data[train_data.homeOwner != owner_drop[1]]
	train_data=train_data[train_data.homeOwner != owner_drop[2]]
# New features
	train_data.openCreditLineRatio=train_data.numOpenCredit/train_data.numTotalCredit
	train_data.amountPayPerMonth=train_data.loanFund/train_data.numPayment
	train_data.loan2balance=train_data.loanFund/train_data.creditBalance
	train_data.loan2income=train_data.loanFund/train_data.income

	test_data.openCreditLineRatio=test_data.numOpenCredit/train_data.numTotalCredit
	test_data.amountPayPerMonth=test_data.loanFund/train_data.numPayment
	test_data.loan2balance=test_data.loanFund/train_data.creditBalance
	test_data.loan2income=test_data.loanFund/train_data.income
	train_data.loanReq=train_data.loanFund/train_data.loanReq
	train_data.investFrac=train_data.investFrac/train_data.loanFund
	test_data.loanReq=test_data.loanFund/test_data.loanReq
	test_data.investFrac=test_data.investFrac/test_data.loanFund
	# category to dummy varibles
	category_features=['subGrade','numPayment','homeOwner','category']
	for feature in category_features:
		tmp_feature=pd.get_dummies(train_data[feature],prefix=feature)
		train_data = pd.concat([train_data, tmp_feature], axis=1)
		tmp_feature1=pd.get_dummies(test_data[feature],prefix=feature)
		test_data = pd.concat([test_data, tmp_feature1], axis=1)
	train_data=train_data.drop(category_features,axis=1)
	test_data=test_data.drop(category_features,axis=1)
	## drop features
	drop_features=['employer','zipcode','incomeVeri','loanTitle','reason','grade']
	drop_features2=['yearsEmployed','numDerogatory','state','listStatus']
	drop_features=drop_features+feature_missing+drop_features2
	train_data=train_data.drop(drop_features,axis=1)
	test_data=test_data.drop(drop_features,axis=1)
# fill missing
	for x in train_data.columns:
		if (train_data[x].dtype==float):
			train_data[x].fillna(train_data[x].mean(),inplace=True)
	for x in test_data.columns:
		if (test_data[x].dtype==float):
			test_data[x].fillna(test_data[x].mean(),inplace=True)
	test_data=test_data.drop('rate',axis=1)
	test_data.to_csv('featured_Test.csv',sep=',',index=False)
	train_data.to_csv('featured_Train.csv',sep=',',index=False)

if __name__ == "__main__":
    main()