import os
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import numpy as np

mySVM = SVC(kernel='rbf')

def load_data(path):
    os.chdir(path)

    f = open('train_feat.txt','r').readlines()
    tr_feat = [[float(x) for x in f[i].split(' ')] for i in range(len(f))]

    f = open('train_labels.txt','r').readlines()
    tr_lb = [int(x) for x in f]

    f = open('test_feat.txt','r').readlines()
    tst_feat = [[float(x) for x in f[i].split(' ')] for i in range(len(f))]

    f = open('test_labels.txt','r').readlines()
    tst_lb = [int(x) for x in f]

    print('Summary:')
    print('\t # classes = ' + str(max(tst_lb)))
    print('\t Feature dimension = ' + str(len(tr_feat[0])))
    print('\t # train/test samples = ' + str(len(tr_feat)) + '/' +str(len(tst_feat)))

    return tr_feat,tr_lb,tst_feat,tst_lb

def main():
    #path = os.environ['HOME']+'/Acads/AML'
    path = os.getcwd()
    [tr_feat,tr_lb,tst_feat,tst_lb ] = load_data(path)
    
    C_range = np.logspace(1,4,4)
    Gamma_range = np.logspace(1,4,4)
    param_grid = dict(gamma=Gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=32)
    clf = GridSearchCV(mySVM, param_grid=param_grid, cv=cv, n_jobs=-1)
    clf.fit(tr_feat,tr_lb)
    print("The best parameters are %s with a score of %0.2f"
      % (clf.best_params_, clf.best_score_))

    pred = clf.predict(tst_feat) - np.array(tst_lb)
    count = 0
    for i in pred:
        if i==0:
            count = count+1
    print('Accuracy = ' + str(count*100/len(tst_feat)))

if __name__ == '__main__':
    main()
