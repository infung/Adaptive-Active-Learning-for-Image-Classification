from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

def gauNBtrain(Lset):
    X_train = Lset[:, :-1]
    y_train = Lset[:, -1]
    bayes = GaussianNB()
    bayes.fit(X_train, y_train)
    w = 0
    return bayes, w

def gauNB(Tset, bayes):
    X_test = Tset[:, :-1]
    y_test = Tset[:, -1]
    y_pred = bayes.predict(X_test)
    accu = accuracy_score(y_test, y_pred)
    prob = bayes.predict_proba(X_test)
    prob_log = bayes.predict_log_proba(X_test)
    y_vec = np.argmax(prob.transpose(), axis=0)
    return accu, y_vec, prob, prob_log

