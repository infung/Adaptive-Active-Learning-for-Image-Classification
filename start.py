import numpy as np
from scipy.io import loadmat
from Adaptive_learning.activeLearning import activecombsample

# Load the data from MATLAB files
# data = loadmat('voc07_densesift.mat')
# voc07_densesift = data['feats']
#
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

# labelset = np.loadtxt('labelset.csv', delimiter=',')
# unlabelset = np.loadtxt('unlabelset.csv', delimiter=',')
# testset = np.loadtxt('testset.csv', delimiter=',')


repeatnum = 100  # Selection iterations
allbeta = np.arange(0, 1.1, 0.1)

# Method
print('Adaptive Active Learning begins...')
adaptiveresult, _ = activecombsample(labelset, unlabelset, testset, allbeta, repeatnum)
