# Adaptive Active Learning for Image Classification

### Intruduction ###

* This is a group project of the HKU course COMP7404 Machine Learning.
* Group members:
    - Feng Yan
    - Ying Yuanyuan
* This project is the implementation of the paper **"Xin Li, Y. Guo, Adaptive Active Learning for Image Classification, CVPR 2013"** .

### Prerequisite ###
* before executing the "experiment.py" file, you need to install modelAL:
```
pip install modAL-python
```

### How to run ###
* If you want to test the accuracy of the proposed approach with the provided dataset, just simply execute the "start.py" file.

* If you want to try out other approaches like 'Most Uncertainty', 'Fixed Combination' or the proposed approach using a fixed beta, just simply execute the "experiment.py" file with specified parameters.

* If you want to try out the 'Caltech101' dataset:
    - Download the Caltech101 from https://data.caltech.edu/records/mzrjq-6wc02 
    - Run the jupyter file step by step, and finally a caltech101.mat file will be produced which contains the labeld set, unlabeled set and test set with label as the last column.
