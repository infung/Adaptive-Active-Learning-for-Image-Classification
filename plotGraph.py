import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat

from Adaptive_learning.activeLearning import activecombsample
from Adaptive_learning.otherApproaches import otherApproaches

# data = loadmat('voc_labelset.mat')
# labelset = data['labelset']
#
# data = loadmat('voc_unlabelset.mat')
# unlabelset = data['unlabelset']
#
# data = loadmat('voc_testset.mat')
# testset = data['testset']

data = loadmat('caltech_101.mat')
labelset = data['labelset']
unlabelset = data['unlabelset']
testset = data['testset']

repeatnum = 100

#           1 - single beta with proposed approach
#           2 - most uncertainty
#           3 - fixed combination with cosine distance to measure an information density
approach = {
    'method': 2,
    'num': 1  # only useful for method 2, select the top 'num' uncertainty
}
beta = 0.5
print('most uncertainty approach')
adaptiveresult, _ = otherApproaches(labelset, unlabelset, testset, beta, repeatnum, approach)
accuracy_uncertainty = adaptiveresult

print('fixed combination approach')
approach['method'] = 3
beta = 1
adaptiveresult, _ = otherApproaches(labelset, unlabelset, testset, beta, repeatnum, approach)
accuracy_fixedComb = adaptiveresult

allbeta = np.arange(0, 1.1, 0.1)
print('proposed approach')
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
plt.savefig('Figure_1')
plt.show()

print('proposed approach with fixed beta')
approach['method'] = 1
accuracy_beta = []
betaList = [0.25, 0.5, 0.75, 1]

for b in betaList:
    adaptiveresult, _ = otherApproaches(labelset, unlabelset, testset, b, repeatnum, approach)
    accuracy_beta.append(adaptiveresult)

plt.plot(iteration, accuracy_proposed, label='Proposed Approach')
plt.plot(iteration, accuracy_beta[0], label='β = 0.25')
plt.plot(iteration, accuracy_beta[1], label='β = 0.5')
plt.plot(iteration, accuracy_beta[2], label='β = 0.75')
plt.plot(iteration, accuracy_beta[3], label='β = 1')
plt.xlabel('Number of Labeled Instances')
plt.ylabel('Classification Accuracy')
plt.title('Logistic Regression Classification Accuracy')
plt.legend()
plt.savefig('Figure_2')
plt.show()
