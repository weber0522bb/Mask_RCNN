from sklearn import svm
from sklearn import calibration
import numpy as np

def svm_classify(x, y):
    '''
    FUNC: train SVM classifier with input data x and label y
    ARG:
        - x: input data, HOG features
        - y: label of x, face or non-face
    RET:
        - clf: a SVM classifier using sklearn.svm. (You can use your favorite
               SVM library but there will be some places to be modified in
               later-on prediction code)
    '''
    #########################################
    ##          you code here              ##
    #########################################
    print('Training...')
    clf = svm.LinearSVC(C=0.05)
    clf = calibration.CalibratedClassifierCV(clf, method='sigmoid', cv=5)
    clf.fit(x, y)
    #########################################
    ##          you code here              ##
    #########################################

    return clf
