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
* If you want to test the accuracy of the proposed approach with the provided dataset (very small subset, only 29 training data), just simply execute the "start.py" file.

* If you want to try out other approaches like 'Most Uncertainty', 'Fixed Combination' or the proposed approach using a fixed beta, just simply execute the "experiment.py" file with specified parameters.

* **To dear TA**:
    - The 'voc07' datasets are just small subset from VOC2007 dataset, even smaller than what we used to conduct the accuracy graph in the paper. It's a good dataset for you to test out our code.
    - If you want to try out the 'Caltech101' dataset:
         * Download the Caltech101 from https://data.caltech.edu/records/mzrjq-6wc02 
