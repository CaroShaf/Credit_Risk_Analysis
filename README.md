# Credit Risk Analysis
Financial institutions everywhere are in the business of loaning money to individuals, families and businesses for many important reasons such as home mortgages or business 
startup costs.  Financial institutions use credit scores as a measure of the risk of default on loans.  Although the majority of loans are profitable to the lending institutions 
through repayment of the principal with interest, there is a percentage that turn out to be very costly when they result in defaults.  The purpose of this analysis is to determine
if six different machine learning algorithms are reliable predictors of a loan applicant's risk of default.

Resources used to conduct this study:  Python 3.7, Pandas, Numpy, Jupyter Notebook, Scikit-learn, Imbalanced-learn, data from LendingTree.com (CSV format)

## Overview of the loan prediction risk analysis
The use of machine learning algorithms in financial lending institutions is not new.  In the ever-expanding field of artificial intelligence, the minimization of risk and time
required to produce accurate results can be enhanced.  In this study, we examined six different supervised machine learning models to classify risk into high or low, that is,
loan approval or not.  Unsupervised and regression models were not included.

### Preprocessing data
Before any data can be run through a learning model, careful attention must be paid to datatypes, nulls, as well as all relevant data features.  Text was converted to numerical
data to classify the values for analysis.  Pandas was the tool of choice to complete the data munging phase of this study.

### Imbalanced data

the model will be much better at predicting non-fraudulent transactions than fraudulent ones. This is a problem if the goal is to detect fraudulent transactions!

Because the majority of loans do not default, the data set is very heavy on low-risk applicants with a small percentage of high-risk.  So, it was important to consider the data
set to be imbalanced and select algorithms that use methods to address the imbalance.  The following are the models that were selected for this study with a description of the
sampling algorithm selected to address the imbalanced classes.  The logistic regression models predict the probability, based on available data, of classification into binary
classes.  In this case, the classes are low- and high-risk.  To address the class imbalance, the four logistic regression models (with their indicated sampling algorithms) were
run both with stratification of split test and training dataset present and without.  Although the stratification did not result in improvement of balanced accuracy score in all
four cases, it did improve the SMOTE oversampling case significantly while the others were either unchanged or decreased.  The values presented in the Results selection below 
are those with the stratify in place for all four logistic regression studies.

The two models that are not logistic regression models belong to the class of ensemble models.  An ensemble model is like it indicates.  It uses an ensemble, or a combination,
of multiple other models to achieve a higher accuracy.  A random forest model is an ensemble of decision tree models.  These ensemble learners use bootstrap aggregation and boosting, respectively, to attemptto minimize the problem that ensemble models have with overfitting.  

1. Logistic regression with naive random oversampling
2. Logistic regression with SMOTE oversampling
3. Logistic regression with cluster centroids undersampling
4. Logistic regression with SMOTEENN combination over- and under-sampling
5. Balanced Random Forest Classifier (ensemble learner)
6. Easy Ensemble AdaBoost Classifier (ensemble learner)

## Results
Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all six machine learning models. Use screenshots of your outputs to support your results.

ACCURACY SCORES
Imagine the following scenario, in which a credit card company wishes to detect fraudulent transactions in real time. Historically, out of 100,000 transactions, 10 have been fraudulent. An analyst writes a program to detect fraudulent transactions, but due to an uncaught bug, it flags every transaction as not fraudulent. Out of 100,000 transactions, it correctly classifies the 99,990 transactions that are not fraudulent, and it erroneously classifies all 10 transactions that are fraudulent.
The program's accuracy score appears to be impressive at 99.99%. However, it fails spectacularly at its job, detecting 0 out of 10 fraudulent transactions, a success rate of 0%.

PRECISION
Precision = TP/(TP + FP)

RECALL
Sensitivity = TP/(TP + FN)
perfect recall means everyone that is classed as high-risk will default, and everyone that is classed as low-risk will not default

In summary, there's a fundamental tension between precision and sensitivity. Highly sensitive tests and algorithms tend to be aggressive, as they do a good job of detecting the intended targets, but also risk resulting in a number of false positives. High precision, on the other hand, is usually the result of a conservative process, so that predicted positives are likely true positives; but a number of other true positives may not be predicted.

F1 SCORE
2(Precision * Sensitivity)/(Precision + Sensitivity)
To illustrate the F1 score, let's return to the scenario of a faulty algorithm for detecting fraudulent credit card transactions. Say that 100 transactions out of 100,000 are fraudulent.
In such a scenario, the sensitivity is very high, while the precision is very low. Clearly, this is not a useful algorithm. Nor does averaging the sensitivity and precision yield a useful figure. Let's try calculating the F1 score. The F1 score is 0.002. We noted previously that there's usually a trade-off between sensitivity and precision, and that a balance must be struck between the two. A useful way to think about the F1 score is that a pronounced imbalance between sensitivity and precision will yield a low F1 score.

## Summary
Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.

SVM model comparasion to logistic model / look at saved confusion table
Scaling/normalization could be a factor

In such a case, even a model that blindly classifies every transaction as non-fraudulent will achieve a very high degree of accuracy. As we saw previously, one strategy to deal with class imbalance is to use appropriate metrics to evaluate a model's performance, such as precision and recall.
