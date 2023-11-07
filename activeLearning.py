import numpy as np
from scipy.linalg import pinv
from sklearn.naive_bayes import GaussianNB
from Adaptive_learning.entropyandDensity import getEntropy, getInfoDensity, predictionLoss, calInvSubmat
from Adaptive_learning.gaussianKernel import gaussianKernel

from Adaptive_learning.logRegNew import logRegNew, logregtrain


def activecombsample(Lset, Uset, Tset, beta, repeatnum):
    # Set the initial values for the regularization parameter lambda and the kernel width sigma
    sigma = 2

    # Get the number of instances and features in the unlabeled set Uset
    ut, d = Uset.shape
    d = d - 1  # Subtract 1 to exclude the label column

    # Get the number of instances in the labeled set Lset and determine the unique classes in the test set Tset
    lt = Lset.shape[0]
    classes = np.unique(Tset[:, -1])

    # Compute the Gaussian kernel matrix uK for the unlabeled set
    uK = gaussianKernel(Uset[:, :-1], Uset[:, :-1], sigma)

    # Initialize the list of unlabeled instances ulist and the inverse of the unlabeled kernel matrix invU
    ulist = np.arange(ut)
    U = uK
    invU = pinv(U)

    # Perform the initial evaluation by training a logistic regression classifier on the labeled set Lset
    # and evaluating its accuracy on the test set Tset
    logReg, w = logregtrain(Lset)
    accu, _, _, _ = logRegNew(Tset, logReg)
    accu_result = [accu]
    _, _, prob, log_proba = logRegNew(Uset, logReg)

    #
    # bayes = GaussianNB()
    # bayes = bayes.fit(Lset[:,0:-1],Lset[:,-1])
    # prob = bayes.predict_proba(Uset[:,0:-1])
    # log_proba = bayes.predict_log_proba(Uset[:,0:-1])

    # Enter a loop that iterates repeatnum times (excluding the initial evaluation)
    for iter in range(2, repeatnum+2) :
        index = 0
        hb = []
        # Compute the uncertainty measure (entropy) for the candidate instances
        for i in Uset[:,0:-1]:
            # compute entropy
            entropy = getEntropy(prob[index], log_proba[index])
            # compute information density
            info = getInfoDensity([index + 1], U, invU)
            # combine them together with different value of beta
            cur_hb = []
            for b in beta:
                cur_hb.append(entropy ** b * info ** (1 - b))
            hb.append(cur_hb)
            index = index + 1
        S = np.argmax(hb, axis=0)
        S = list(set(S)) # get rid of  redundancy
        # compute the cost for each candidate instance and select the one with the minimum cost
        rows = np.array([Uset[i] for i in S])
        _, _, prob, _ = logRegNew(rows, logReg)
        costs = []
        for idx, s in enumerate(S):
            expecterr = 0
            for idy, y in enumerate(classes):
                # add unlabeled data into labeled data
                tempLdata = np.vstack((Lset, np.concatenate((Uset[s, :-1], np.array([y])))))
                # retrain on the augmented set L+<x,y>,
                templogReg, _ = logregtrain(tempLdata)
                tempUdata = np.copy(Uset)
                tempUdata = np.delete(tempUdata, s, axis=0)
                # predict on unlabled set
                _, yvec_temp, probnew, _ = logRegNew(tempUdata, templogReg)
                t = np.arange(tempUdata.shape[0])
                p_yhat = predictionLoss(t, yvec_temp, probnew)
                expecterr += prob[idx, idy] * np.sum(1 - p_yhat)
            costs.append(expecterr)
        id = np.argmin(costs)
        real_id = S[id]
        # Move the selected instance from the unlabeled set to the labeled set
        ut -= 1
        lt += 1
        Lset = np.concatenate((Lset, np.array([Uset[real_id]])), axis=0)
        Uset = np.delete(Uset, real_id, axis=0)
        ulist = np.delete(ulist, real_id, axis=0)
        invU = calInvSubmat(U, invU, real_id)
        U = uK[ulist, :][:, ulist]

        # Retrain the logistic regression classifier on the updated labeled set
        # and evaluate its accuracy on the test set
        logReg, w = logregtrain(Lset)
        accu, _, _, _ = logRegNew(Tset, logReg)
        accu_result.append(accu)
        _, _, prob, log_proba = logRegNew(Uset, logReg)

    # Set the variable selections to contain the selected instances from the labeled set
    selections = Lset[-repeatnum:, :]

    return accu_result, selections