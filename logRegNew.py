from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

def logregtrain(Lset):
    X_train = Lset[:, :-1]
    y_train = Lset[:, -1]
    logReg = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000)
    logReg.fit(X_train, y_train)
    w = logReg.coef_
    return logReg, w

def logRegNew(Tset, logReg):
    X_test = Tset[:, :-1]
    y_test = Tset[:, -1]
    y_pred = logReg.predict(X_test)
    accu = accuracy_score(y_test, y_pred)
    prob = logReg.predict_proba(X_test)
    prob_log = logReg.predict_log_proba(X_test)
    y_vec = np.argmax(prob.transpose(), axis=0)
    return accu, y_vec, prob, prob_log
