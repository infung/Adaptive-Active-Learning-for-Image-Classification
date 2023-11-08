# Xin Li, Y. Guo, Adaptive Active Learning for Image Classification, CVPR 2013

import numpy as np
from scipy.linalg import pinv
from sklearn.naive_bayes import GaussianNB
from Adaptive_learning.entropyandDensity import getEntropy, getInfoDensity, predictionLoss, calInvSubmat
from Adaptive_learning.gaussianKernel import gaussianKernel
from Adaptive_learning.logRegNew import logRegNew, logregtrain
from modAL.density import information_density


def otherApproaches(Lset, Uset, Tset, beta, repeatnum, approach):
    # input:
    #       beta:   a float
    #       approach.method:   1 - single beta with proposed approach
    #                          2 - most uncertainty
    #                          3 - fixed combination with cosine distance to measure an information density

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

    # Initialize the list of unlabeled information density
    if approach['method'] == 3:
        info_density = information_density(Uset, 'cosine')

    # Enter a loop that iterates repeatnum times (excluding the initial evaluation)
    for iter in range(2, repeatnum+2):
        # fit on the Unlabeled set to get the distribution
        _, _, prob, log_proba = logRegNew(Uset, logReg)
        instance_to_move = 0
        if approach['method']== 1:   # fixed beta
            index = 0
            hb = []
            # Compute the uncertainty measure (entropy) for the candidate instances
            for i in Uset[:, 0:-1]:
                # compute entropy
                entropy = getEntropy(prob[index], log_proba[index])
                # compute information density
                info = getInfoDensity([index + 1], U, invU)
                # combine them together with a fixed beta
                hb.append(entropy ** beta * info ** (1 - beta))
                index = index + 1
            # Get the index of maximum combination value
            instance_to_move = np.argmax(hb)

        elif approach['method'] == 2:     # most uncertainty
            index = 0
            # Calculate entropy vector and sort it in ascending order
            entropies = []
            for i in Uset[:, 0:-1]:
                # compute entropy
                entropy = getEntropy(prob[index], log_proba[index])
                entropies.append(entropy)
                index = index + 1
            eids = np.argsort(entropies)[::-1]
            # Select the top n uncertainty
            topIds = eids[:approach['num']]
            # compute the cost for each candidate instance and select the one with the minimum cost
            rows = np.array([Uset[i] for i in topIds])
            _, _, prob, _ = logRegNew(rows, logReg)
            costs = []
            for idx, s in enumerate(topIds):
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
            instance_to_move = topIds[id]

        elif approach['method'] == 3:      # fixed combination with beta = 1
            index = 0
            hb = []
            # Compute the uncertainty measure (entropy) for the candidate instances
            for i in Uset[:, 0:-1]:
                # compute entropy
                entropy = getEntropy(prob[index], log_proba[index])
                # get the corresponding info_density
                info = info_density[index]
                # combine them together with beta = 1
                hb.append(entropy * info)
                index = index + 1
            # Get the index of maximum combination value
            instance_to_move = np.argmax(hb)

        # Move the selected instance from the unlabeled set to the labeled set
        ut -= 1
        lt += 1
        Lset = np.concatenate((Lset, np.array([Uset[instance_to_move]])), axis=0)
        Uset = np.delete(Uset, instance_to_move, axis=0)
        ulist = np.delete(ulist, instance_to_move, axis=0)
        invU = calInvSubmat(U, invU, instance_to_move)
        U = uK[ulist, :][:, ulist]
        #  evaluate its accuracy on the test set
        logReg, w = logregtrain(Lset)
        accu, _, _, _ = logRegNew(Tset, logReg)
        accu_result.append(accu)

    # Set the variable selections to contain the selected instances from the labeled set
    selections = Lset[-repeatnum:, :]

    return accu_result, selections