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
The results of the six tested models are mixed and largely unimpressive.  As we progressed through the different types of sampling with linear regression classifiers and on to 
the ensemble classifiers with more sophisticated balancing and minimization techniques for overfitting, we do see improved balanced accuracy scores.  But these scores alone 
don't tell us everything we need to know.

This chart contains a summary of the balanced accuracy scores as well as the classification report for each of the six models:

https://github.com/CaroShaf/Credit_Risk_Analysis/blob/main/images/comparisongrid.png

Using the confusion matrices and classification reports (individual images are linked below for each model) for each test (all with random_sample=1, to restrict the dataset to 
comparable results across models) the following key trends can be observed.

BALANCED ACCURACY SCORES
As mentioned above, as we traversed through the six models, the balanced accuracy scores improved from a low of .523 (the Linear Regression model with clustered centroid 
undersampling) to the best of .932 for the Easy Ensemble Classifier.  The balanced accuracy score is simply the average of recall obtained on both classes, so if one of the two
classes has an especially low recall, the overall balanced accuracy score could still appear to be fairly decent when the majority class scores with perfect recall.  However, in
these cases, the recall for both classes are fairly close and thus track fairly closely with the balanced accuracy scores.

PRECISION TP/(TP + FP)
As can be seen in the chart above, all models (all with random_sample=1, to restrict the dataset to comparable results across models) were able to predict with 100% certainty 
that low-risk applicants were low-risk. Yet all four logistic regression models only gave about a 1% probability of correctly classifying a high-risk application as a high-risk, 
as there were many false positives (high-risk customers classified as low-risk).  The ensemble models did slightly better at 3% and 9% respectively, but again false positives 
would necessarily be high in order for the overall precision to be low.  Basically, as great as these models do at predicting low-risk customers being low-risk of defaulting, 
they outright fail when it comes to this very important group, high-risk customers classed as low-risk and thus with a high likelihood of defaulting.  

RECALL
Since recall/sensitivity is calculated by TP/(TP + FN) and our numbers of false negatives are consistently low throughout the different models, the recall/sensitivity in all but 
one case is greater than 60%.  Again, recall doesn't seem to tell much about the performance of our models since both the number of true positives (as above in precision) and 
false negatives are low compared to false positives and true negatives.  Perfect recall means everyone that is classed as high-risk will default, and everyone that is classed as 
low-risk will not default.  Our numbers indicate that our models don't predict either of these groups very well, and as such the relative numbers are low even though relative to 
each other, their sensitivity appears greater than 60% in all cases.

The fundamental tension between precision and sensitivity is at work here. Highly sensitive tests and algorithms tend to be aggressive, as they do a good job of detecting the 
intended targets, but also risk resulting in a number of false positives.  We have more false positives and true negatives than anything else. High precision, on the other hand, 
is hampered severely in these cases by the large number of false positives and the low number of true positives.

F1 SCORE  2(Precision * Sensitivity)/(Precision + Sensitivity)
In all over our models, the sensitivity is relatively high, while the precision is extremely low for the minority class resulting in very low F1 scores. A pronounced imbalance
in precision and sensitivity will show up in the F1 score and it most definitely does for all of our minority class across models.

REFERENCE IMAGES
Links to images of each balanced accuracy score calculation and classification report:

Linear Regression with Random Oversampling details [https://github.com/CaroShaf/Credit_Risk_Analysis/blob/main/images/LRwRO.png]
Linear Regression with SMOTE Oversampling details [https://github.com/CaroShaf/Credit_Risk_Analysis/blob/main/images/LRwSO.png]
Linear Regression with Cluster Centroids Undersampling details [https://github.com/CaroShaf/Credit_Risk_Analysis/blob/main/images/LRwCCU.png]
Linear Regression with SMOTEENN Over- and Under- sampling details [https://github.com/CaroShaf/Credit_Risk_Analysis/blob/main/images/LRwSOU.png]
EasyEnsembleClassifier [https://github.com/CaroShaf/Credit_Risk_Analysis/blob/main/images/EEC.png]
BalancedRandomForestClassifier [https://github.com/CaroShaf/Credit_Risk_Analysis/blob/main/images/BRF.png]

## Summary
If we are to choose between the six models presented here, we would have to cautiously go with the EasyEnsembleClassifier because of its balanced accuracy score, recall and F1
score. However, it is not much better than any of the others when it comes to precision.  This is a weak recommendation, though, and other models could be considered.  We did
look at Support Vector Machines and it compared favorably with the Logistic Regression models.

A lot more could be done to tune this particular model or to explore other models.  No scaling/normalization was used in this study, so that could potentially enhance precision. 
There are many features that could be eliminated or edited.  In the BalancedRandomForestClassifier, it was shown that the most relevant feature was the total amount of the loan.
Some of the other higher ranking features relied on a customer's past banking history.  Those factors may show an unintended and damaging bias in predictions.

If we are to consider bias, then training data is a part of the problem. If we rely on data that comes from existing biased processes and datasets, it will teach the model to be
biased too. Testing may also be problematic, as common practice is to keep back some of the same biased training data to use for testing the system, which would obviously fail 
to show up any bias issues.  As a new user of machine learning algorithms, I feel the gravity of the possibility of contributing to a system which is harmful to some classes of
people.  At the same time, the models don't do a great job on predicting default anyway, and as such, aren't even protecting lending agencies from harm either.  It is important
to know which variables are being considered in these credit scoring models and how the variables are affecting people’s scores in addition to the effect that lack of precision
has on the lending institutions.

