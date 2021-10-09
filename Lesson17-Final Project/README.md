# Final Project

## Part 1 (Initial results):

Try to run the empty one first, it will generate the initial results.

```
tester.dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()
```
```
python poi_id.py  
```

### Different Classifiers:
```
GaussianNB:
GaussianNB()
	Accuracy: 0.25560	Precision: 0.18481	Recall: 0.79800	F1: 0.30011	F2: 0.47968
	Total predictions: 10000	True positives: 1596	False positives: 7040	False negatives:  404	True negatives:  960

SVC:
SVC()
	Accuracy: 0.79920	Precision: 0.16667	Recall: 0.00100	F1: 0.00199	F2: 0.00125
	Total predictions: 10000	True positives:    2	False positives:   10	False negatives: 1998	True negatives: 7990

Decision Tree:
DecisionTreeClassifier()
	Accuracy: 0.69210	Precision: 0.23619	Recall: 0.24150	F1: 0.23881	F2: 0.24042
	Total predictions: 10000	True positives:  483	False positives: 1562	False negatives: 1517	True negatives: 6438

Kmeans
KMeans(n_clusters=2)
	Accuracy: 0.77540	Precision: 0.20290	Recall: 0.04200	F1: 0.06959	F2: 0.04992
	Total predictions: 10000	True positives:   84	False positives:  330	False negatives: 1916	True negatives: 7670

Total no of records: 146
Total no of POIs: 18
Total no of POIs: 128
```

### Question 1:
Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

**Goal:** To classify POI out of enron email dataset. Precision and Recall are greater than 0.3 at least. Machine learnig can create different models to classify and predict dataset. The dataset has multiple features, POI can be related to some important features. However, the dataset also had outliers, we can retrain to remove points with largest residual errors (10%).

Recall is pretty impressive, but accuracy and precisions are pathetically low. Apparantly, Salary is not the only feature useful to identify POIs effectively.

## Part 2 (Final Results)
```
Now Printing Features Priority
deferred_income 2
total_stock_value 1
expenses 3
poi_mail_ratio 4
[2 1 3 4]
Decision Tree:
DecisionTreeClassifier(min_samples_split=6)
	Accuracy: 0.81631	Precision: 0.38602	Recall: 0.32850	F1: 0.35494	F2: 0.33859
	Total predictions: 13000	True positives:  657	False positives: 1045	False negatives: 1343	True negatives: 9955
```
### Question 2:
What features did you end up using in your POI identifier, and what selection process did you use to pick them? 
Did you have to do any scaling? Why or why not? As part of the assignment, 
you should attempt to engineer your own feature that does not come ready-made in the dataset -- 
explain what feature you tried to make, and the rationale behind it. 
(You do not necessarily have to use it in the final analysis, only engineer and test it.) 
In your feature selection step, if you used an algorithm like a decision tree, 
please also give the feature importances of the features that you use, 
and if you used an automated feature selection function like SelectKBest, 
please report the feature scores and reasons for your choice of parameter values.
[relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

### Question 3:
What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]


### Question 4:
What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]


### Question 5:
What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]


### Question 6:
Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

```
features_list = ['poi', 'total_payments', 'exercised_stock_options', 'shared_receipt_with_poi','deferred_income', 
'total_stock_value', 'expenses', 'poi_mail_ratio']
```
```
Now Printing Features Priority
total_payments 1
exercised_stock_options 3
shared_receipt_with_poi 1
deferred_income 2
total_stock_value 1
expenses 1
poi_mail_ratio 1
[1 3 1 2 1 1 1]
Decision Tree:
DecisionTreeClassifier(min_samples_split=6)
	Accuracy: 0.83408	Precision: 0.45641	Recall: 0.41100	F1: 0.43252	F2: 0.41934
	Total predictions: 13000	True positives:  822	False positives:  979	False negatives: 1178	True negatives: 10021
```
