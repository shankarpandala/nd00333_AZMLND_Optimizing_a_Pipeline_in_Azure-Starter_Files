# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
### Problem Description

The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

### Solution

The best performing model was a VotingEnsemble ran by AutoML with an accuracy of 91.68%"

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

PythonScriptStep was used to create a training pipeline which calls the "train.py" script. Pipeline is provided with deatils of the compute cluster to run and other arguments such as input and output location.

TabularDatasetFactory is used to load the dataset.
The data was cleaned and pre-processed with a function for
  - dropping null valus
  - One-hot encoding

After splitting the data into train and test set, A Logistic Regression model was fit on the training data

The hyperparameters optimized are:
 - Regularization Strength(C)
 - Max iterations(max_iter)

Hyperdrive is used to find the best hyperparameters that gives the best accuracy

**What are the benefits of the parameter sampler you chose?**

RandomParameterSampling is a technique which randomly picks the hyperparameters from the range provided (both discrete and continous) and check if the selected random hyperprameters gives best accuracy. This sampling techniques converges faster than the gird search method

**What are the benefits of the early stopping policy you chose?**

Bandit policy is based on slack factor and evaluation interval. Bandit terminates runs where the primary metric is not within the specified slack factor compared to the best performing run. Slack factor is the slack allowed with respect to the best performing training run.

## AutoML
**Below are the models ran by AutoML
****************************************************************************************************
ITERATION: The iteration being evaluated.

PIPELINE: A summary description of the pipeline being evaluated.

DURATION: Time taken for the current iteration.

METRIC: The result of computing score on the fitted pipeline.

BEST: The best observed score thus far.
****************************************************************************************************

    ITERATION PIPELINE                                      DURATION      METRIC      BEST

         0   MaxAbsScaler LightGBM                          0:00:40       0.9152    0.9152
         1   MaxAbsScaler XGBoostClassifier                 0:00:41       0.9153    0.9153
         2   MaxAbsScaler RandomForest                      0:00:30       0.8957    0.9153
         3   MaxAbsScaler RandomForest                      0:00:36       0.8880    0.9153
         4   MaxAbsScaler RandomForest                      0:00:31       0.8006    0.9153
         5   MaxAbsScaler RandomForest                      0:00:44       0.7612    0.9153
         6   SparseNormalizer XGBoostClassifier             0:00:52       0.9116    0.9153
         7   MaxAbsScaler GradientBoosting                  0:00:46       0.9024    0.9153
         8   StandardScalerWrapper RandomForest             0:00:33       0.9005    0.9153
         9   MaxAbsScaler LogisticRegression                0:00:35       0.9083    0.9153
        10   MaxAbsScaler ExtremeRandomTrees                0:02:09       0.8880    0.9153
        11   SparseNormalizer XGBoostClassifier             0:00:45       0.9121    0.9153
        12   MaxAbsScaler LightGBM                          0:00:31       0.8910    0.9153
        13   MaxAbsScaler LightGBM                          0:00:41       0.9046    0.9153
        14   SparseNormalizer XGBoostClassifier             0:01:54       0.9124    0.9153
        15   StandardScalerWrapper LightGBM                 0:00:40       0.8952    0.9153
        16   StandardScalerWrapper RandomForest             0:00:52       0.8880    0.9153
        17   SparseNormalizer XGBoostClassifier             0:00:48       0.9142    0.9153
        18   MaxAbsScaler LightGBM                          0:00:43       0.9084    0.9153
        19   SparseNormalizer XGBoostClassifier             0:00:44       0.9125    0.9153
        20   SparseNormalizer XGBoostClassifier             0:00:56       0.9121    0.9153
        21   SparseNormalizer LightGBM                      0:00:32       0.9053    0.9153
        22   SparseNormalizer LightGBM                      0:00:36       0.9120    0.9153
        23   MaxAbsScaler LightGBM                          0:00:35       0.9098    0.9153
        24   SparseNormalizer LightGBM                      0:00:42       0.9141    0.9153
        25   StandardScalerWrapper XGBoostClassifier        0:00:31       0.8880    0.9153
        26   StandardScalerWrapper XGBoostClassifier        0:00:41       0.9081    0.9153
        27   SparseNormalizer XGBoostClassifier             0:01:08       0.9137    0.9153
        28   SparseNormalizer LightGBM                      0:00:49       0.9105    0.9153
        29   SparseNormalizer XGBoostClassifier             0:01:27       0.9098    0.9153
        30   SparseNormalizer XGBoostClassifier             0:00:30       0.8880    0.9153
        31   StandardScalerWrapper XGBoostClassifier        0:00:30       0.8959    0.9153
        32   StandardScalerWrapper LightGBM                 0:00:41       0.9079    0.9153
        33   StandardScalerWrapper LightGBM                 0:00:30       0.8918    0.9153
        34                                                  0:03:01          nan    0.9153
        35   VotingEnsemble                                 0:01:11       0.9168    0.9168
        36   StackEnsemble                                  0:01:27       0.9148    0.9168
Stopping criteria reached at iteration 37. Ending experiment.

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

Difference between AutoML and Hyperdrive pipelines accuracy is very narrow. AutoML has edge over Hyperdrive as we are optimising only Logistic Regression and its very limited  hyperparameters where as AutoML has option to choos lot of algoritms and hyperparameters.

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

Hyperprameter optimization is performed only for Logistic Regression there is a lot of scope to optimise hyperparameters in other algorithms. All other potential algorithms needs to be tried out.