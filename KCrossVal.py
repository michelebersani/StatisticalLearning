from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy.linalg import norm
from sklearn.model_selection import RepeatedKFold
import time



# function that train the algorithm with the function passed
def kFoldCross(trainCallback, predictCallback, X_dev, Y_dev, n_splits, err_func):
    kf = KFold(n_splits=n_splits)
    ValError = []
    TrError = []

    start_time = time.process_time()
    for train, val in kf.split(X_dev):
        trainCallback(X_dev[train], Y_dev[train])

        val_predicted = predictCallback(X_dev[val])
        val_e = err_func(val_predicted, Y_dev[val])
        ValError.append(val_e)

        tr_predicted = predictCallback(X_dev[train])
        tr_e = err_func(tr_predicted, Y_dev[train])
        TrError.append(tr_e)
    seconds = time.process_time()-start_time

    ValError = np.array(ValError)
    TrError = np.array(TrError)

    mean_ValError = np.mean(ValError)
    std_ValError = np.std(ValError)

    mean_TrError = np.mean(TrError)

    return mean_ValError, std_ValError, mean_TrError, seconds


def MeanAbsError(out: np.ndarray, target: np.ndarray):
    return np.abs(out - target).mean()

def BinaryAccuracy(out: np.ndarray, target: np.ndarray):
    out = (out > 0).astype(int)
    target = (target > 0).astype(int)
    return np.abs(out - target).mean()