---
title: "Prediction of Manner of Weight Lifting Exercises"
author: "Kun Yao"
date: "05/21/2015"
output: html_document
---

**Summary**

In this report we develop practical machine learning algorithms to qualify how well people do weight lifting exercises, based on sensor data from accelerometers on the belt, forearm, arm, and dumbbell. An estimate of error rate for the fitted algorithms will also be given.

**Data**

We first load the necessary library and raw data (courtesy of <http://groupware.les.inf.puc-rio.br/har>) into R.

```r
library(ggplot2)
library(caret)
rawData <- read.csv("pml-training.csv")
```

There are total 160 columns in the raw data. 36 columns are the raw data collected from the four sensors which detect three-axes acceleration, gyroscope  and  magnetometer. Also the 3 calculated Euler angles for each sensor should be important. Most of the remaining columns are the various statistics (total/average/variance etc) of the above data, which also contain many NA's. We leave only the total acceleration columns, which always contain valid data. The final dataset that will be used for our prediction is

```r
newData <- rawData[grepl("_[xyz]$|^total|^roll|^pitch|^yaw|classe", names(rawData))]
```

Here we pick the Euler angles on two sensors (belt and dumbbell) and plot them with the classe variable, it looks like that certain clustering does exist. Machine learning should be able to help us identifying the patterns.

![plot of chunk unnamed-chunk-3](figure/unnamed-chunk-3-1.png) 

In order to cross-validate several prediction models and estimate out-of-sample 
error, we further split the training set into two subsets.

```r
set.seed(0)
inTrain <- createDataPartition(newData$classe, p=0.7, list=FALSE)
training <- newData[inTrain,]
testing <- newData[-inTrain,]
```


**Prediction with Classification Model**

Both random forest and boosting models are among the most popular machine learning techniques. Here we will build and test two models, random forest (rf) and stochastic gradient boosting (gbm), using R's default train control method.


```r
library(randomForest)
rfFit <- train(classe~., method="rf", data=training)
print(rfFit)
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9878877  0.9846738  0.002071481  0.002623235
##   27    0.9881142  0.9849604  0.001985916  0.002517627
##   52    0.9774727  0.9714945  0.005610254  0.007097742
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
library(gbm)
boostFit <- train(classe~., method="gbm", data=training, verbose=FALSE)
print(boostFit)
```

```
## Stochastic Gradient Boosting 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
## 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa      Accuracy SD
##   1                   50      0.7487443  0.6814013  0.007456333
##   1                  100      0.8165478  0.7677391  0.005955708
##   1                  150      0.8494765  0.8094550  0.004989106
##   2                   50      0.8515566  0.8118688  0.005374035
##   2                  100      0.9013296  0.8751140  0.004467629
##   2                  150      0.9262934  0.9067299  0.004525918
##   3                   50      0.8930458  0.8645846  0.005327301
##   3                  100      0.9361905  0.9192592  0.003172658
##   3                  150      0.9554405  0.9436276  0.003212191
##   Kappa SD   
##   0.009386356
##   0.007446073
##   0.006193138
##   0.006734766
##   0.005580560
##   0.005685638
##   0.006716651
##   0.003979200
##   0.004027233
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```

As we can see, with optimal tuning of both models, they archived good accuracy (0.989 for random forest versus 0.955 for boosting). It is also worth noting that the tuning in random forest model does not help too much, the three mtry values (2, 27 and 52) all result in accuracy higher than 97%. We can now apply them to predict out testing sets for cross-validation.


```r
table(predict(rfFit, testing), testing$classe)
```

```
##    
##        A    B    C    D    E
##   A 1669    6    0    0    0
##   B    4 1128    1    0    1
##   C    1    4 1018    9    1
##   D    0    1    7  954    3
##   E    0    0    0    1 1077
```

```r
sum(predict(rfFit, testing) == testing$classe) / length(testing$classe)
```

```
## [1] 0.993373
```

```r
table(predict(boostFit, testing), testing$classe)
```

```
##    
##        A    B    C    D    E
##   A 1644   33    0    0    2
##   B   16 1077   28    4   14
##   C    6   29  978   27   11
##   D    2    0   20  926    7
##   E    6    0    0    7 1048
```

```r
sum(predict(boostFit, testing) == testing$classe) / length(testing$classe)
```

```
## [1] 0.9639762
```

Both models show a higher accuracy rate than obtained during the training process. And if we choose random forest algorithm, we can estimate the out-of-sample error rate to be 1-0.993373=0.67%. This is probably an optimistic estimate due to overfitting by hand picking the better fitted model from two classification models.

We also examine the predictor importance for our fitted random forest model, as shown in the following variance important plot. The top three predictors are roll_belt, pitch_forearm and yaw_belt.

![plot of chunk unnamed-chunk-7](figure/unnamed-chunk-7-1.png) 

In the next plot we also visually checked our prediction on the testing set against the roll_betl and pitch_forearm predictors. The false predictions appears to be rather randomly distributed, except that maybe a bit more concentration in the high pitch_forearm regime (> 75).


```r
predRight <- predict(rfFit, testing)==testing$classe
qplot(roll_belt, pitch_forearm, color=predRight, data=testing, main="testing predictions")
```

![plot of chunk unnamed-chunk-8](figure/unnamed-chunk-8-1.png) 

**Conclusion**

We studied the prediction of manner of weight lifting exercises via two machine learning algorithms, random forest and boosting. Random forest model yields an excellent estimated error rate of only 0.67%. It is not very sensitive to overfitting of mtry parameter. We will choose our random forest algorith to predict on real testing.


```r
testData <- read.csv("pml-testing.csv")
answers <- predict(rfFit, testData)
```
