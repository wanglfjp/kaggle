##https://github.com/terrychenism/NeuralNetTests/blob/master/Kaggle/kaggle-higgs/graph_train.py
import csv
import sys
sys.path.append('../../python/')
import numpy as np
import scipy as sp
import xgboost as xgb
import sklearn.cross_validation as cv
import pandas as pd

def AMS(s, b):
    '''
    Approximate median significance:
        s = true positive rate
        b = false positive rate
    '''
    assert s >= 0
    assert b >= 0
    bReg = 10.
    return np.sqrt(2.0 * ((s + b + bReg) * np.log(1 + s / (b + bReg)) - s))


def get_rates(prediction, solution, weights):
    '''
    Returns the true and false positive rates.
    This assumes that:
        label 's' corresponds to 1 (int)
        label 'b' corresponds to 0 (int)
    '''
    assert prediction.size == solution.size
    assert prediction.size == weights.size

    # Compute sum of weights for true and false positives
    truePos  = sum(weights[(solution == 1) * (prediction == 1)])
    falsePos = sum(weights[(solution == 0) * (prediction == 1)])

    return truePos, falsePos
def momentum_xyz(particleName,df):
    pt = df[particleName+'pt']
    eta =df[particleName+'eta']
    phi =df[particleName+'phi']
    px=pt*np.cos(phi)
    py=pt*np.sin(phi)
    tantheta2=np.exp(-eta)
    pz=pt*(1+tantheta2*tantheta2)/2/tantheta2
    ptot=px*px+py*py+pz*pz
    missing = (pt<-0)
    px[missing]=-999.0
    py[missing]=-999.0
    pz[missing]=-999.0
    ptot[missing]=-999.0
    return (pd.DataFrame({particleName+'px' : px,
                      particleName+'py' : py,
                      particleName+'pz' : pz,
                      particleName+'ptot': ptot}))
def get_momentum_features(df):
    lep = momentum_xyz('PRI_lep_',df)
    jet_leading = momentum_xyz('PRI_jet_leading_',df)
    jet_subleading = momentum_xyz('PRI_jet_subleading_',df)
    tau = momentum_xyz('PRI_tau_',df)
    return (lep.join(tau).join(jet_leading).join(jet_subleading))

def add_momentum_features(df):
    return df.join(get_momentum_features(df)).replace([np.inf, -np.inf], np.nan).fillna(-999.)

def momentum_xyz(particleName,df):
    pt = df[particleName+'pt']
    eta =df[particleName+'eta']
    phi =df[particleName+'phi']
    px=pt*np.cos(phi)
    py=pt*np.sin(phi)
    tantheta2=np.exp(-eta)
    pz=pt*(1+tantheta2*tantheta2)/2/tantheta2
    ptot=px*px+py*py+pz*pz
    missing = (pt<-0)
    px[missing]=-999.0
    py[missing]=-999.0
    pz[missing]=-999.0
    ptot[missing]=-999.0
    return (pd.DataFrame({particleName+'px' : px,
                      particleName+'py' : py,
                      particleName+'pz' : pz,
                      particleName+'ptot': ptot}))
def get_momentum_features(df):
    lep = momentum_xyz('PRI_lep_',df)
    jet_leading = momentum_xyz('PRI_jet_leading_',df)
    jet_subleading = momentum_xyz('PRI_jet_subleading_',df)
    tau = momentum_xyz('PRI_tau_',df)
    return (lep.join(tau).join(jet_leading).join(jet_subleading))

def add_momentum_features(training_file):
#    dpath = 'C:\wanglf2016\kaggle\ATLAS\data'
#    df_train = pd.read_csv(dpath+'/training.csv',nrows=10000)
    df_train = pd.read_csv(training_file)  #,nrows=10000)
#    df_test = pd.read_csv(dpath+'/test.csv',nrows=1000)
    return get_momentum_features(df_train)
#    return df.join(get_momentum_features(df)).replace([np.inf, -np.inf], np.nan).fillna(-999.)

def get_training_data(training_file):
    '''
    Loads training data.
    '''
    data = list(csv.reader(open(training_file, "rb"), delimiter=','))
    X       = np.array([map(float, row[1:-2]) for row in data[1:]])
    labels  = np.array([int(row[-1] == 's') for row in data[1:]])
    weights = np.array([float(row[-2]) for row in data[1:]])
    newfeature=add_momentum_features(training_file)
    Y=np.concatenate((X,(newfeature.values)),axis=1)
    return Y, labels, weights

def get_test_data(testing_file):
    '''
    Loads testing data.
    '''
    data = list(csv.reader(open(testing_file, "rb"), delimiter=','))
    X       = np.array([map(float, row[1:]) for row in data[1:]])
#    labels  = np.array([int(row[-1] == 's') for row in data[1:]])
#    weights = np.array([float(row[-2]) for row in data[1:]])
    testId = np.array([row[0] for row in data[1:]])
    newfeature=add_momentum_features(testing_file)
    Y=np.concatenate((X,(newfeature.values)),axis=1)
    return (Y, testId)
def get_accuracy(prediction, solution):
    '''
    Returns the true and false positive rates.
    This assumes that:
        label 's' corresponds to 1 (int)
        label 'b' corresponds to 0 (int)
    '''
    assert prediction.size == solution.size

    # Compute sum of weights for true and false positives
    accuracy=1.0*sum([1 for i, j in zip(prediction, solution) if i == j])/len(solution)
    T11=sum([1 for i, j in zip(prediction, solution) if i==1 and j==1])  #TT
    T12=sum([1 for i, j in zip(prediction, solution) if i==0 and j==1])  #Nn
    T21=sum([1 for i, j in zip(prediction, solution) if i==1 and j==0])  #FP
    T22=sum([1 for i, j in zip(prediction, solution) if i==0 and j==0])   #TF
	# print('True positive rate= %f'%(1.0*T11/(T21+T11)))
	# print('Flase positive rate= %f'%(1.0*T12/(T21+T11)))
	# print('True Negative rate= %f'%(1.0*T22/(T12+T22)))
	# print('Flase Negative rate= %f'%(1.0*T12/(T12+T22)))
	# print('Accuracy = %f'%(accuracy_train))
    precision=1.0*T11/(T21+T11)
    FP=1.0*T21/(T21+T11)
    TN=1.0*T22/(T12+T22)
    FN=1.0*T12/(T12+T22)
    return (accuracy,precision,FP,TN,FN,T11,T12,T21,T22)

def estimate_performance_xgboost(training_file, param, num_round, folds):
    '''
    Cross validation for XGBoost performance
    '''
    # Load training data
    f=open("summary_bst_scan.txt","a")
    X, labels, weights = get_training_data(training_file)

    # Cross validate
    kf = cv.KFold(labels.size, n_folds=folds, random_state=4234)
    npoints  =30
    # Dictionary to store all the AMSs
    ams_best_idall_AMS = {}
    all_AMS = {}
    summay_table=[]
    for curr in range(npoints):
        all_AMS[curr] = []
    # These are the cutoffs used for the XGBoost predictions
    cutoffs  = sp.linspace(0.10, 0.40, npoints)
    for train_indices, test_indices in kf:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = labels[train_indices], labels[test_indices]
        w_train, w_test = weights[train_indices], weights[test_indices]

        # Rescale weights so that their sum is the same as for the entire training set
        w_train *= (sum(weights) / sum(w_train))
        w_test  *= (sum(weights) / sum(w_test))

        sum_wpos = sum(w_train[y_train == 1])
        sum_wneg = sum(w_train[y_train == 0])

        # construct xgboost.DMatrix from numpy array, treat -999.0 as missing value
        xgmat = xgb.DMatrix(X_train, label=y_train, missing=-999.0, weight=w_train)

        # scale weight of positive examples
        param['scale_pos_weight'] = sum_wneg / sum_wpos
        # you can directly throw param in, though we want to watch multiple metrics here
        plst = param.items()#+[('eval_metric', 'ams@0.15')]

        watchlist = []#[(xgmat, 'train')]
        bst = xgb.train(plst, xgmat, num_round, watchlist)

        # Construct matrix for test set
        xgmat_test = xgb.DMatrix(X_test, missing=-999.0)
        y_out = bst.predict(xgmat_test)
        res  = [(i, y_out[i]) for i in xrange(len(y_out))]
        rorder = {}
        for k, v in sorted(res, key = lambda x:-x[1]):
            rorder[k] = len(rorder) + 1

        # Explore changing threshold_ratio and compute AMS
        best_AMS = -1.
        k_best=0
        for curr, threshold_ratio in enumerate(cutoffs):
            y_pred = sp.zeros(len(y_out))
            ntop = int(threshold_ratio * len(rorder))
            for k, v in res:
                if rorder[k] <= ntop:
                    y_pred[k] = 1

            truePos, falsePos = get_rates(y_pred, y_test, w_test)
            this_AMS = AMS(truePos, falsePos)
            all_AMS[curr].append(this_AMS)
            if this_AMS > best_AMS:
                best_AMS = this_AMS
                k_best=curr
## estimate the accuracy
            accu,precision,FP,TN,FN,T11,T12,T21,T22=get_accuracy(y_pred, y_test)
            summay_table.append([accu,precision,FP,TN,FN,T11,T12,T21,T22])
#            f.write("accu,precision,FP,TN,FN")
#             f.write('\n')
#             f.write("accu=%f.4,precision=%f.4,FP=%f.4,TN=%f.4,FN=%f.4"%(accu,precision,FP,TN,FN))
#             print ("accu=%f.4,precision=%f.4,FP=%f.4,TN=%f.4,FN=%f.4"%(accu,precision,FP,TN,FN))
#             f.write('\n')
# #            f.write('T11,T12,T21,T22')
#             f.write('T11=%f,T12=%f,T21=%f,T22=%f'%(T11,T12,T21,T22))
#             f.write('\n')

        print ("Best AMS =", best_AMS)
        f.write("Best AMS =%f"%(best_AMS))
        f.write('\n')

    print "------------------------------------------------------"
    ams_best_id=0
    ams_max=0
    import numpy as np
    summay_table_mean=[]
    n_para=9
    for j in range(npoints):
        idx_j=[j+npoints*i for i in range(folds)]
        avg=[]
        for k in range(n_para):
            avg_temp=np.mean([summay_table[i][k] for i in idx_j])
            avg.append(avg_temp)
        summay_table_mean.append(avg)
    print summay_table_mean
    for curr, cut in enumerate(cutoffs):
        print "Thresh = %.4f: AMS = %.4f, std = %.4f" % \
            (cut, sp.mean(all_AMS[curr]), sp.std(all_AMS[curr]))
        print "Accuracy = %.4f: Precision = %.4f, FP = %.4f" % \
            (summay_table_mean[curr][0], summay_table_mean[curr][1],\
             summay_table_mean[curr][2])
        f.write('\n')
        f.write("Thresh = %.4f: AMS = %.4f, std = %.4f" % \
                (cut, sp.mean(all_AMS[curr]), sp.std(all_AMS[curr])))
        f.write('\n')
        f.write("Accuracy = %.4f: Precision = %.4f, FP = %.4f" % \
            (summay_table_mean[curr][0], summay_table_mean[curr][1],\
             summay_table_mean[curr][2]))
        f.write('\n')
        f.write("Confusing matrix")
        f.write('\n')
        f.write("TP=%.0f  FN=%.0f" %(summay_table_mean[curr][5], \
                                     summay_table_mean[curr][6]))
        f.write('\n')
        f.write("FP=%.0f  TN=%.0f" %(summay_table_mean[curr][7],\
                                     summay_table_mean[curr][8]))
        f.write('\n')
        if ams_max<sp.mean(all_AMS[curr]):
            ams_max=sp.mean(all_AMS[curr])
            ams_best_id=curr
    print "------------------------------------------------------"
    f.write('\n')
    f.write('The best solution:\n')
    f.write("Thresh = %.4f: AMS = %.4f, std = %.4f" % \
                (cutoffs[ams_best_id], sp.mean(all_AMS[ams_best_id]), sp.std(all_AMS[ams_best_id])))
    f.write('\n')

    f.close()


def Model_performance_xgboost(training_file, testing_file,param, num_round, threshold_ratio_finall):
#def Model_performance_xgboost(training_file,testing_file, param, num_round, folds):
    '''
    Trainging validation for XGBoost performance
    '''
    # Load training data
#    f=open("summary_training.txt","a")
    f=open("summary_bst_scan.txt","a")
    f.write('-------**************---------')
    f.write('\n')
    f.write('Finall compuation on Training and test data only')
    f.write('\n')
    X, labels, weights = get_training_data(training_file)
    X_test, testID = get_test_data(testing_file)
    # Cross validate
 #    kf = cv.KFold(labels.size, n_folds=folds)
    npoints  = 40
    # Dictionary to store all the AMSs
    all_AMS = {}
    for curr in range(npoints):
        all_AMS[curr] = []
    # These are the cutoffs used for the XGBoost predictions
    cutoffs  = sp.linspace(0.08, 0.18, npoints)
    cutoffs  = sp.linspace(0.14, 0.16, npoints)
    trainingonly=1
    if(trainingonly==1):
        X_train= X
        y_train= labels
        w_train = weights

        # Rescale weights so that their sum is the same as for the entire training set
        w_train *= (sum(weights) / sum(w_train))

        sum_wpos = sum(w_train[y_train == 1])
        sum_wneg = sum(w_train[y_train == 0])

        # construct xgboost.DMatrix from numpy array, treat -999.0 as missing value
        xgmat = xgb.DMatrix(X_train, label=y_train, missing=-999.0, weight=w_train)

        # scale weight of positive examples
        param['scale_pos_weight'] = sum_wneg / sum_wpos
        # you can directly throw param in, though we want to watch multiple metrics here
        plst = param.items()#+[('eval_metric', 'ams@0.15')]

        watchlist = []#[(xgmat, 'train')]
        bst = xgb.train(plst, xgmat, num_round, watchlist)

        y_out = bst.predict(xgmat)
        res  = [(i, y_out[i]) for i in xrange(len(y_out))]
        rorder = {}
        for k, v in sorted(res, key = lambda x:-x[1]):
            rorder[k] = len(rorder) + 1

        # Explore changing threshold_ratio and compute AMS
        best_AMS = -1.
        for curr, threshold_ratio in enumerate(cutoffs):
            y_pred = sp.zeros(len(y_out))
            ntop = int(threshold_ratio * len(rorder))
            for k, v in res:
                if rorder[k] <= ntop:
                    y_pred[k] = 1

            truePos, falsePos = get_rates(y_pred, y_train, w_train)
            this_AMS = AMS(truePos, falsePos)
            all_AMS[curr].append(this_AMS)
            if this_AMS > best_AMS:
                best_AMS = this_AMS
        print "Best AMS =", best_AMS
        f.write("Best AMS =%f"%(best_AMS))
        f.write('\n')
    print "------------------------------------------------------"
    for curr, cut in enumerate(cutoffs):
        print "Thresh = %.5f: AMS = %.4f, std = %.4f" % \
            (cut, sp.mean(all_AMS[curr]), sp.std(all_AMS[curr]))
        f.write("Thresh = %.5f: AMS = %.4f, std = %.4f" % \
                (cut, sp.mean(all_AMS[curr]), sp.std(all_AMS[curr])))
        f.write('\n')
    print "------------------------------------------------------"
    f.close()
#    X_test, testID
    xgmat_test = xgb.DMatrix(X_test, missing=-999.0)
    y_out = bst.predict(xgmat_test)
    res  = [(i, y_out[i]) for i in xrange(len(y_out))]
    rorder = {}
    for k, v in sorted(res, key = lambda x:-x[1]):
        rorder[k] = len(rorder) + 1
    y_pred = sp.zeros(len(y_out))
    ntop = int(threshold_ratio_finall * len(rorder))
    outfile = 'higgs_pred.csv'
    fo = open(outfile, 'w')
    nhit = 0
    ntot = 0
    fo.write('EventId,RankOrder,Class\n')
    for k, v in res:
        if rorder[k] <= ntop:
            y_pred[k] =1
            lb = 's'
            nhit += 1
        else:
            y_pred[k] =0
            lb = 'b'
        fo.write('%s,%d,%s\n' % ( testID[k],  len(rorder)+1-rorder[k], lb) )
        ntot += 1
    fo.close()
    print ('finished writing into prediction file')

def main():
    # setup parameters for xgboost
    param = {}
    # use logistic regression loss, use raw prediction before logistic transformation
    # since we only need the rank
# solution 1--------------------------------
    param['objective'] = 'binary:logitraw'
    param['silent'] = 1
    param['nthread'] = 1
    param['eval_metric'] = 'rmse' #'auc'
#    param['bst:eta'] = 0.1
    param['bst:max_depth'] = 6
    param['min_child_weight']=1
    param['gamma']=0
    param['subsample']=1    # = c(0.5, 0.75, 1),
    param['colsample_bytree']=0.9  # = c(0.6, 0.8, 1))
    param['learning_rate']=0.2
    num_round = 200 # Number of boosted trees
# solution 1 end--------------
 #### Solutions 2
# xgb1_best_model = XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.3,
#        gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=6,
#        min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
#        objective='binary:logistic', reg_alpha=0, reg_lambda=1,
#        scale_pos_weight=1, seed=0, silent=True, subsample=1)
#     param['objective'] = 'binary:logitraw'
#     param['silent'] = 1
#     param['nthread'] = 1
#     param['eval_metric'] = 'auc'
# #    param['bst:eta'] = 0.1
#     param['bst:max_depth'] = 6
#     param['min_child_weight']=1
#     param['gamma']=0
#     param['subsample']=1    # = c(0.5, 0.75, 1),
#     param['colsample_bytree']=0.3  # = c(0.6, 0.8, 1))
#     num_round = 140 # Number of boosted trees
#

    folds = 5 # Folds for CV
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
    max_depth_values=[3,5,6,7]
    min_child_weight_values=[1,3,5]
    # max_depth_values=[5]
    # min_child_weight_values=[4]
    #
    # for x in max_depth_values:
    #     for y in min_child_weight_values:
    #         f=open("summary_bst_scan.txt","a")
    #         f.write('\n')
    #         f.write('-------------')
    #         f.write('\n')
    #         param['max_depth']=x
    #         param['min_child_weight']=y
    #         f.write('param {}={}'.format('max_depth',x))
    #         f.write('\n')
    #         f.write('param {}={}'.format('mini_child_weight',y))
    #         f.write('\n')
    #         f.write('--------------')
    #         f.write('\n')
    #         f.close()
    #         estimate_performance_xgboost("C:/wanglf2016/kaggle/ATLAS/data/training.csv", param, num_round, folds)
   # gamma_vlaues=[10,100]
    # for x in gamma_vlaues:
    #         f=open("summary_bst_scan.txt","a")
    #         f.write('\n')
    #         f.write('-------------')
    #         f.write('\n')
    #         param['gamma']=x
    #         f.write('param {}={}'.format('gamma',x))
    #         f.write('\n')
    #         f.write('--------------')
    #         f.write('\n')
    #         f.close()
    # subsample_values=[0.6,0.8,0.9,1]
    # colsample_bytree_values=[0.6,0.8,0.9,1]
    # subsample_values=[1]
    # colsample_bytree_values=[0.5,0.7]
    # for x in subsample_values:
    #     for y in colsample_bytree_values:
    #         f=open("summary_bst_scan.txt","a")
    #         f.write('\n')
    #         f.write('-------------')
    #         f.write('\n')
    #         param['subsample']=x
    #         param['colsample_bytree']=y
    #         f.write('param {}={}'.format('subsample',x))
    #         f.write('\n')
    #         f.write('param {}={}'.format('colsample_bytree',y))
    #         f.write('\n')
    #         f.write('--------------')
    #         f.write('\n')
    #         f.close()

    # learning_rate_values =[0.3,0.2,0.18,0.16,0.15,0.14,0.13,0.12]
    # # num_round = 120
    # num_round_01=120
    # for x in learning_rate_values:
    #         f=open("summary_bst_scan.txt","a")
    #         f.write('\n')
    #         f.write('-------------')
    #         f.write('\n')
    #         param['learning_rate'] =x
    #         f.write('param {}={}'.format('learning_rate',x))
    #         f.write('\n')
    #         f.write('--------------')
    #         f.write('\n')
    #         f.close()
    num_round_values =[175,180,185,190]
    num_round_values =[170,173,178]
    num_round_values =[170,171,172]
    # for x in num_round_values:
    #         f=open("summary_bst_scan.txt","a")
    #         f.write('\n')
    #         f.write('-------------')
    #         f.write('\n')
    #         num_round =x
    #         f.write('num_round={}'.format(x))
    #         f.write('\n')
    #         f.write('--------------')
    #         f.write('\n')
    #         f.close()
#    estimate_performance_xgboost("C:/wanglf2016/kaggle/ATLAS/data/training.csv", param, num_round, folds)
    training_file="C:/wanglf2016/kaggle/ATLAS/data/training.csv"
    testing_file="C:/wanglf2016/kaggle/ATLAS/data/test.csv"
    threshold_ratio_finall=0.01536
    Model_performance_xgboost(training_file, testing_file,param, num_round, threshold_ratio_finall)
if __name__ == "__main__":
    main()

