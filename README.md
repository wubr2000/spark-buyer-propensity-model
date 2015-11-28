Buyer's Lead Prediction Model v.1.0
===================

Below is the documentation for the Buyer's Lead Prediction Model v.1.0. 

## Purpose

This set of programs allows users to:

1. Train (or retrain) the logistic regression models necessary for predicting the likelihood (propensity score) that an RDC user will submit a buyer's lead using DART/DFP data.

2. Predict propensity scores for each user from new DART data.


## Getting Started

1. Launch Spark cluster on AWS of appropriate nodes. You can follow the instructions for [launching a Spark cluster](https://wiki.move.com/display/BI/Steps+to+Launching+an+Apache+Spark+Cluster+on+AWS).

2. [to be filled in]


## To Make Predictions

To make predictions, run the `predict.py` file from the command line. You must run this program in a **Pyspark shell**. 

```
$ /root/spark/bin/pyspark predict.py [full pathname and filenames of DART raw log]
```

The first argument passed to the `predict.py` is the file name(s) of the DART raw log for which predictions are to be made (along with full pathname). The DART raw log file(s) must reside in an S3 bucket. Below is an example of inputting the DART raw logs for the entire day of 9/10/2015 for prediction.

```
$ /root/spark/bin/pyspark predict.py s3n://move-data-engineering-p1/data/events/dart/current/*20150910*
```

Once run, `predict.py` will invoke `dart_model_main.py` which comprises the actual logic of running the saved prediction models to calculate a propensity score for the new set of users.

The outputs of `predict.py` are 1) the logistic regression models used, and 2) a list of tuples containing users (in the form of JY tags) and their respective propensity scores to submit a buyer's lead. Here is an example of these tuples:

```
[(u'xyz7091e9f2c425f96e9183822a5da5c', 0.5),
 (u'9582123154cc4672a784c1e5e0297411', 0.4087282200918592),
 (u'51b20d1sf3b3432e8ce94c9262b1bd60', 0.5),
 (u'fa1d33dfgdfg456sd887d088d60f201a', 0.9084670550171319),
 (u'546fgeg45sdg56s4d8309ca27b418682', 0.5),
 (u'sdreh009df0dfg30a433f42389b52d04', 0.5),
 (u's90g908sd098sd098sf56c2f3009affb', 0.5),
 (u'sdg890sdg908dsg0a00667fafa8673f1', 0.5),
 (u'c921ca6eeb77474f93d0b0f58f474a44', 0.6047080180604159),
 (u'23f9dwhgwwedg9bbbf73513weggwe1b9', 0.23638238057955482),]
```

To see the outputs of the `predict.py` file in the command line. Do the following:

```
$ models, predictions = `/root/spark/bin/pyspark predict.py [full pathname and filenames of DART raw log]`
$ echo $predictions
```

You can then either use these predictions directly or save them to a file for later use. 


## To Train/Retrain Models

Use the `train.py` file to train a set of models for all metros (specified in the `train.py` file) and one "national" model (where 80% of all users regardless of metros were used for training). These training models will be saved under the `models` folder. **Before training/retraining, please make sure that the `model_prefix` specified in the `train.py` file is correct. You might accidentally overwrite old models if you used the same model prefix.** While model training is taking place, command line prompts appear intermitently to let you know which metro is currently being trained.

To train models, run the `train.py` file from the command line. You must run this program in a **Pyspark shell**. 

```
$ /root/spark/bin/pyspark train.py [full pathname and filenames of DART raw log]
```

The first argument passed to the `train.py` is the file name(s) of the DART raw log for which predictions are to be made (along with full pathname). The DART raw log file(s) must reside in an S3 bucket. Below is an example of inputting the DART raw logs for the entire day of 9/10/2015 for training the logistic models.

```
$ /root/spark/bin/pyspark train.py s3n://move-data-engineering-p1/data/events/dart/current/*20150910*
```



