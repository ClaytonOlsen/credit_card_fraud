# <a title="Credit Card Fraud Detection with ML Models"> Credit Card Fraud Detection with ML Models</a>

## Introduction

It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. In this project, we aim to classify credit card purchases as fraudulent or non-fraudulent based on credit card purchase data found on [kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud). We will fit a variety of ML models to see what performs best based on criterion that favors decreasing false negatives as they are most crucial in fraud detection with imbalanced data.

## Data Overview

The credit card fraud dataset contains post principal component analysis variables to maintain anonimity as well as the amount paid for each purchase, the time, and the response variable the class (1 for fraduelent and 0 for non-fradulent) displayed below.

<img src="https://github.com/ClaytonOlsen/credit_card_fraud/blob/101e4b25c206f462f5ac41026f38e38987536ac8/images/Screen%20Shot%202021-09-09%20at%203.50.47%20PM.png" width="600" height="400" alt="Computer Hope">

## Exploratory Analysis

Other than the transaction and the amount, we do not have information on what the other columns are because of privacy. We do know that the PCA columns are scaled already, so we will need to scale the time/amount variables.

The distribution of the classes are very skewed with much more non-fradulent cases than fradulent cases as shown in the following image.

<img src="https://github.com/ClaytonOlsen/credit_card_fraud/blob/101e4b25c206f462f5ac41026f38e38987536ac8/images/fraudvnotfraud.pdf" width="600" height="400" alt="fvnf">


The first thing we must do is gather a basic sense of our data. Remember, except for the transaction and amount we dont know what the other columns are (due to privacy reasons). The only thing we know, is that those columns that are unknown have been scaled already.
















