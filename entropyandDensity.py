import numpy as np
import copy
from scipy.io import loadmat

def predictionLoss(row, col, prob_new):
    indexes = np.ravel_multi_index((row, col), prob_new.shape)
    p_yhat = []
    i = 0
    for row in prob_new:
        for item in row:
            if i in indexes:
                p_yhat.append(item)
            i += 1
    p_yhat = np.array(p_yhat)
    return p_yhat

def calInvSubmat(B, invB, id):
    t = len(B[0])
    u = [0 for i in range(t)]
    u[0] = -1
    u = np.asarray(u)
    u = np.resize(u,[t,1])
    oids = list(range(1,id)) + list(range(id+1,t+1))
    v = [B[id-1][id-1]-1]
    for oid in oids:
        v.append(B[id-1][oid-1])
    v = np.asarray(v)
    v = np.transpose([v])
    v = np.resize(v,[1,t])

    w = [0]
    order = [id]
    for oid in oids:
        w.append(B[oid-1][id-1])
        order.append(oid)
    w = np.asarray(w)
    w = np.resize(w,[t,1])

    invA = np.zeros((t,t))
    for i in range(t):
        for j in range(t):
            invA[i][j] = invB[order[i]-1][order[j]-1]

    invA1 = invA - (invA@u@v@invA)/(1+v@invA@u)
    invA2 = invA1 - (invA1@w@u.T@invA1)/(1+u.T@invA1@w)
    invBi = invA2[1:,1:]
    return invBi


def getEntropy(prob, log_proba):
    ans = 0
    for i in range(len(prob)):
        ans += -prob[i]*log_proba[i]
    return ans


def getInfoDensity(yids,K, invK):
    n = len(K)
    yn = len(yids)

    SigY = np.diag(K)
    tmp = np.zeros([yn,1])
    for i in range(yn):
        tmp[i] = SigY[yids[i]-1]
    SigY = copy.deepcopy(tmp)
    SigYU = copy.deepcopy(tmp)
    for i in range(yn):
        id = yids[i]
        uid = list(range(1,id)) + list(range(id+1,n+1))
        invSigUU = calInvSubmat(K,invK,id)
        uid = [i-1 for i in uid]

        K1 = K[id-1][uid]
        K1 = np.resize(K1,[1,n-1])
        K2 = np.zeros([n-1,1])
        index = 0
        for u in uid:
            K2[index]=K[u][id-1]
            index += 1
        SigYU[i] = SigYU[i] - K1@invSigUU@K2;

    minfo = 0.5*np.log2(SigY/SigYU)
    minfo = minfo.item()
    return minfo