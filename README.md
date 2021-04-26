# Credit Risk Analysis
Many financial institutions everywhere are in the business of loaning money to individuals, families and businesses for many important reasons such as home mortgages or business 
startup costs.  They use credit scores as a measure of the risk of default on loans.  Although the large majority of loans are profitable to the lending institutions 
through repayment of the principal with interest, there is a percentage that turn out to costly when they result in defaults.  The purpose of this analysis is to
determine if six different machine learning algorithms are reliable predictors of a loan applicant's risk of default.

Resources used to conduct this study:  Python 3.7, Pandas, Numpy, Jupyter Notebook, Scikit-learn, Imbalanced-learn, data from LendingClub (CSV format)

## Overview of the loan prediction risk analysis
The use of machine learning algorithms in financial lending institutions is not new.  In the ever-expanding field of artificial intelligence, the minimization of risk and time
required to produce accurate results can be enhanced.  In this study, we examined six different supervised machine learning models to classify risk into high or low, that is,
loan approval or not.  Unsupervised and linear regression models were not included.

### Preprocessing data
Before any data can be run through a learning model, careful attention must be paid to datatypes, nulls, as well as all relevant data features.  Text was converted to numerical
data to classify the values for analysis.  Pandas was the tool of choice to complete the data munging phase of this study.

### Imbalanced data
Because the majority of loans do not default, the data set is very heavy on low-risk applicants with a small percentage of high-risk.  A learning algorithm trained with an
unbalanced data set will be great at predicting the low-risk applicants but not so good detecting high-risk ones. So, it was important to consider the data set to be imbalanced 
and select algorithms that use methods to address the imbalance.  The following are the models that were selected for this study with a description of the sampling algorithm 
selected to compensate for the imbalanced classes.  

The logistic regression models predict the probability, based on available data, of classification into binary classes.  It is a flexible and accurate machine learning model, 
which is why we are focusing on it in four of the six cases here (with variations in sampling).  In this case, the classes are low- and high-risk.  To address the class 
imbalance, the four instances of logistic regression models were run both with stratification of split test and training datasets present and without.  Although the 
stratification did not result in improvement of balanced accuracy score in all four cases, it did improve the SMOTE oversampling case significantly while the others were either 
unchanged or decreased.  The values presented in the Results selection below are only those with the stratify in place for all four of the logistic regression studies.

The two models that are not logistic regression models belong to the class of ensemble models.  An ensemble model is like it indicates.  It uses an ensemble, or a combination of 
multiple models to achieve a higher accuracy.  A random forest model is an ensemble of decision tree models.  These ensemble learners use bootstrap aggregation and boosting,
respectively, to attempt to minimize the problem that ensemble models have with overfitting.  

1. Logistic regression with random oversampling - oversamples the minority class by picking samples at random with replacement
2. Logistic regression with SMOTE oversampling (Synthetic Minority Over-sampling Technique) - oversamples minority class by creating new interpolated data from near neighbors
3. Logistic regression with cluster centroids undersampling - majority class centroids created through interpolation of near neighbors, then undersampled to match size of
minority class
4. Logistic regression with SMOTEENN combination over- and under-sampling - oversamples minority class with SMOTE and undersamples majority class with ENN (edited Nearest Neighbors)
5. Balanced Random Forest Classifier (ensemble learner) - undersamples each boostrap sample to balance it
6. Easy Ensemble AdaBoost Classifier (ensemble learner) - trained on different balanced boostrap samples. The balancing is achieved by random undersampling.

## Results
The results of the six tested models are mixed.  As we progressed through the different types of sampling with linear regression classifiers and on to the ensemble classifiers
with more sophisticated balancing and minimization techniques for overfitting, we do see improved balanced accuracy scores.  But these scores alone don't tell us everything we
need to know.  

INSERT COMPARISONS / CONFUSION GRAPHIC

Using the confusion matrices and classification reports for each test (all with random_sample=1, to restrict the dataset to comparable results across models) the following key 
information can be observed.

BALANCED ACCURACY SCORES
Imagine the following scenario, in which a credit card company wishes to detect fraudulent transactions in real time. Historically, out of 100,000 transactions, 10 have been 
fraudulent. An analyst writes a program to detect fraudulent transactions, but due to an uncaught bug, it flags every transaction as not fraudulent. Out of 100,000 transactions, 
it correctly classifies the 99,990 transactions that are not fraudulent, and it erroneously classifies all 10 transactions that are fraudulent.
The program's accuracy score appears to be impressive at 99.99%. However, it fails spectacularly at its job, detecting 0 out of 10 fraudulent transactions, a success rate of 0%.

PRECISION
Precision = TP/(TP + FP)
As can be seen in the spreadsheet, and using a generic confusion matrix as a reference, every model (all with random_sample=1, to restrict the dataset to 
comparable results across models) was able to predict with 100% certainty that low-risk applicants were low-risk. Yet all four logistic regression models only gave about a 1% 
probability of correctly classifying a high-risk application as a high-risk.  The ensemble models did slightly better at 3% and 6% respectively.

RECALL
Sensitivity = TP/(TP + FN)
perfect recall means everyone that is classed as high-risk will default, and everyone that is classed as low-risk will not default

In summary, there's a fundamental tension between precision and sensitivity. Highly sensitive tests and algorithms tend to be aggressive, as they do a good job of detecting the intended targets, but also risk resulting in a number of false positives. High precision, on the other hand, is usually the result of a conservative process, so that predicted positives are likely true positives; but a number of other true positives may not be predicted.

F1 SCORE
2(Precision * Sensitivity)/(Precision + Sensitivity)
To illustrate the F1 score, let's return to the scenario of a faulty algorithm for detecting fraudulent credit card transactions. Say that 100 transactions out of 100,000 are fraudulent.
In such a scenario, the sensitivity is very high, while the precision is very low. Clearly, this is not a useful algorithm. Nor does averaging the sensitivity and precision yield a useful figure. Let's try calculating the F1 score. The F1 score is 0.002. We noted previously that there's usually a trade-off between sensitivity and precision, and that a balance must be struck between the two. A useful way to think about the F1 score is that a pronounced imbalance between sensitivity and precision will yield a low F1 score.

Links to images of each classification report:
INSERT

## Summary
Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.

SVM model comparasion to logistic model / look at saved confusion table
Scaling/normalization could be a factor

In such a case, even a model that blindly classifies every transaction as non-fraudulent will achieve a very high degree of accuracy. As we saw previously, one strategy to deal with class imbalance is to use appropriate metrics to evaluate a model's performance, such as precision and recall.
