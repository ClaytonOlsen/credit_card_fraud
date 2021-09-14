# <a title="Credit Card Fraud Detection with ML Models"> Credit Card Fraud Detection with ML Models</a>

## Introduction

It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. In this project, we aim to classify credit card purchases as fraudulent or non-fraudulent based on credit card purchase data found on [kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud). We will fit a variety of ML models to see what performs best based on criterion that favors decreasing false negatives as they are most crucial in fraud detection with imbalanced data.

## Data Overview

The credit card fraud dataset contains post principal component analysis variables to maintain anonimity as well as the amount paid for each purchase, the time, and the response variable the class (1 for fraduelent and 0 for non-fradulent) displayed below.

<img src="https://github.com/ClaytonOlsen/credit_card_fraud/blob/101e4b25c206f462f5ac41026f38e38987536ac8/images/Screen%20Shot%202021-09-09%20at%203.50.47%20PM.png" width="600" height="350">

## Exploratory Analysis

Other than the transaction and the amount, we do not have information on what the other columns are because of privacy. We do know that the PCA columns are scaled already, so we will need to scale the time/amount variables.

The distribution of the classes are very skewed with much more non-fradulent cases than fradulent cases as shown in the following image.

<img src="https://github.com/ClaytonOlsen/credit_card_fraud/blob/main/images/fraudvnotfraud-1.png" width="500" height="400" data-rotate="90"/>

As all other variables are normalized from the results of PCA, we will explore the relationships with the other variables.

<img src="https://github.com/ClaytonOlsen/credit_card_fraud/blob/main/images/timebyclass-1.png" width="500" height="500" />

#### Correlation Matrices

Looking at the variables correlation with the class varaible we see that V17, V14, V12 and V10 are negatively correlated, meaning the lower these values are, the more likely a purchase will be classfied as fraudulent. Alternatively, V2, V4, V11, and V19 are positively correlated meaning the higher these values are, the mroe likely a purchase will be classified as fraudulent.

<img src="https://github.com/ClaytonOlsen/credit_card_fraud/blob/main/images/correlation_table.png" width="500" height="500" />


## Oversampling

Accuracy can be bias for imbalanced data which classifies all the data points as the prominent class and can achievce near perfect accuracy. We care more about reducing false negatives and increasing True Positives as they are the most important in fraud detection where limited cases are truly limited as fradulent. The Random Forest model offers fairly accurate results, but may be letting to many fradulent cases slip by. Since each tree is built on a 'bag', and each bag is a uniform random sample from the data, each tree will be biased in teh same direction and magnitude, on averafe by class imbalance. Since we want to minimize False Negatives when building a prediction model for the imbalanced data set, we can oversample the smaller class.

<img src="https://github.com/ClaytonOlsen/credit_card_fraud/blob/main/images/Random_Forest.png" width="250" height="250" />

When we want to minimize the False Negatives with imbalanced data, we can oversample the small class (positively classified data in this case).

#### Synthetic Minority Oversampling Technique (SMOTE)

SMOTE works by selecting examples that are close in the feature space, drawing a line between the examples in the feature space and drawing a new sample at a point along that line.

Specifically, a random example from the minority class is first chosen. Then k of the nearest neighbors for that example are found (typically k=5). A randomly selected neighbor is chosen and a synthetic example is created at a randomly selected point between the two examples in feature space. The result is a more balanced dataset as displayed below.

<img src="https://github.com/ClaytonOlsen/credit_card_fraud/blob/main/images/smote_data.png" width="500" height="250" />

With the balanced data we will fit a variety of ML models for credit card fraud classification.


## Model Results
#### Random Forest

<img src="https://github.com/ClaytonOlsen/credit_card_fraud/blob/main/images/random_forest_smote.png" width="500" height="250" />

#### K-Nearest Neighbor

<img src="https://github.com/ClaytonOlsen/credit_card_fraud/blob/main/images/KNN_smote-1.png" width="250" height="250" />

#### Adaboost

<img src="https://github.com/ClaytonOlsen/credit_card_fraud/blob/main/images/adaboost.png" width="500" height="250" />

#### XGBoost

<img src="https://github.com/ClaytonOlsen/credit_card_fraud/blob/main/images/xgboost-1.png" width="500" height="250" />

## Model Comparison

<img src="https://github.com/ClaytonOlsen/credit_card_fraud/blob/main/images/auc-roc-curve.png" width="450" height="250" />

<img src="https://github.com/ClaytonOlsen/credit_card_fraud/blob/main/images/precision-recall_curve.png" width="450" height="250" />


#### Model Criterion

| Model              | Recall Score         | Precision Score      | auc-roc-score      |
| --- | --- | --- | --- |
| Random Forest      | 0.8367346938775511   | 0.7542616021781151   | 0.9182882107541948 |
| K-Nearest Neighbor | 0.8775510204081632   | 0.41716861015865014  | 0.9379401838112846 |
| Adaboost           | 0.9285714285714286   | 0.06073981864221495  | 0.9528285533403007 |
| XGBoost            | 0.9081632653061225   | 0.12469810887183227  | 0.9491576033902588 |


## Summary

Implementing SMOTE on the imbalanced dataset helped us with the imbalance of our labels and provided better recall for our models. All of the models provided accurent overall classification with AUC-ROC over 0.9 which suggests they are all fairly good predictors. In evaluating our models, we put a heavier weight on high recall as miss classifying fradulent casses as not fradulent is the most costly misclassification for credit card fraud. Though, it would be unwise to solely account for the reduction of false negatives as constantly blocking credit card purchases for people who were just making regular purchases may lead to customer complaints and overall dissatisfaction. Therefore, we may not prefer the Adaboost model even though it has the highest recall because of the massive cost to precision as it may be susceptible to overpredicting that credit card purchases are fradulent. We may want to also consider the KNN or XGBoost model as the have more middle ground results while still maintaining strong recall. To further improve our models we may consider doing outlier removal or trying other models that may provide more accurate responses.










