import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat

from Adaptive_learning.activeLearning import activecombsample
from Adaptive_learning.otherApproaches import otherApproaches

data = loadmat('voc_labelset.mat')
labelset = data['labelset']

data = loadmat('voc_unlabelset.mat')
unlabelset = data['unlabelset']

data = loadmat('voc_testset.mat')
testset = data['testset']

# data = loadmat('caltech_101.mat')
# labelset = data['labelset']
# unlabelset = data['unlabelset']
# testset = data['testset']

repeatnum = 20

#           1 - single beta with proposed approach
#           2 - most uncertainty
#           3 - fixed combination with cosine distance to measure an information density
approach = {
    'method': 2,
    'num': 1  # only useful for method 2, select the top 'num' uncertainty
}
beta = 0.5
adaptiveresult, _ = otherApproaches(labelset, unlabelset, testset, beta, repeatnum, approach)
accuracy_uncertainty = adaptiveresult

approach['method'] = 3
beta = 1
adaptiveresult, _ = otherApproaches(labelset, unlabelset, testset, beta, repeatnum, approach)
accuracy_fixedComb = adaptiveresult

allbeta = np.arange(0, 1.1, 0.1)
adaptiveresult, _ = activecombsample(labelset, unlabelset, testset, allbeta, repeatnum)
accuracy_proposed = adaptiveresult

iteration = np.arange(0, repeatnum+1, 1)

plt.plot(iteration, accuracy_proposed, label='Proposed Approach')
plt.plot(iteration, accuracy_uncertainty, label='Most Uncertainty')
plt.plot(iteration, accuracy_fixedComb, label='Fixed Combination')
plt.xlabel('Number of Labeled Instances')
plt.ylabel('Classification Accuracy')
plt.title('Logistic Regression Classification Accuracy')
plt.legend()
plt.show()


approach['method'] = 1
accuracy_beta = []
betaList = [0.25, 0.5, 0.75, 1]
for b in betaList:
    adaptiveresult, _ = otherApproaches(labelset, unlabelset, testset, b, repeatnum, approach)
    labell = 'β = ' + str(b)
    plt.plot(iteration, adaptiveresult, label=labell)
    print(labell)

plt.plot(iteration, accuracy_proposed, label='Proposed Approach')
plt.xlabel('Number of Labeled Instances')
plt.ylabel('Classification Accuracy')
plt.title('Logistic Regression Classification Accuracy')
plt.legend()
plt.show()