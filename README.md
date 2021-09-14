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

<img src="https://github.com/ClaytonOlsen/credit_card_fraud/blob/main/images/correlation_table.png" width="500" height="500" />


## Model Criterion and Oversampling

Accuracy can be bias for imbalanced data which classifies all the data points as the prominent class and can achievce near perfect accuracy. We care more about reducing false negatives and increasing True Positives as they are the most important to avoid in fraud detection where limited cases are truly limited as fradulent.

Rather we will use Recall or True Positive Rate to determine the validity of our models as a decrease in false negatives will increase recall.

<img src="https://github.com/ClaytonOlsen/credit_card_fraud/blob/main/images/Random_Forest.png" width="250" height="250" />

When we want to minimize the False Negatives with imbalanced data, we can oversample the small class (positively classified data in this case).

#### Synthetic Minority Oversampling Technique (SMOTE)

SMOTE works by selecting examples that are close in the feature space, drawing a line between the examples in the feature space and drawing a new sample at a point along that line.

Specifically, a random example from the minority class is first chosen. Then k of the nearest neighbors for that example are found (typically k=5). A randomly selected neighbor is chosen and a synthetic example is created at a randomly selected point between the two examples in feature space. The result is a more balanced dataset as displayed below.

<img src="https://github.com/ClaytonOlsen/credit_card_fraud/blob/main/images/smote_data.png" width="500" height="250" />

The Random Forest results from the SMOTE data is as follows.

<img src="https://github.com/ClaytonOlsen/credit_card_fraud/blob/main/images/random_forest_smote.png" width="500" height="250" />









