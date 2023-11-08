import numpy as np
from scipy.io import loadmat
from Adaptive_learning.activeLearning import activecombsample
from Adaptive_learning.otherApproaches import otherApproaches

# Load the data from MATLAB files
data = loadmat('voc07_densesift.mat')
voc07_densesift = data['feats']

data = loadmat('voc_labelset.mat')
labelset = data['labelset']

data = loadmat('voc_unlabelset.mat')
unlabelset = data['unlabelset']

data = loadmat('voc_testset.mat')
testset = data['testset']

# parameter setting
repeatnum = 10  # Selection iterations
beta = 0.5  # try β ∈ {0.25,0.5,0.75,1}
approach = {
    'method': 3,
    #           1 - single beta with proposed approach
    #           2 - most uncertainty
    #           3 - fixed combination with cosine distance to measure an information density
    'num': 1  # only useful for method 2, select the top 'num' uncertainty
}
approach_name = {
    1: 'Single beta with proposed approach',
    2: 'The most uncertainty',
    3: 'Fixed combination with cosine distance'
}

# Method
print('Active Learning with ', approach_name[approach['method']],' begins...')
adaptiveresult, _ = otherApproaches(labelset, unlabelset, testset, beta, repeatnum, approach)
