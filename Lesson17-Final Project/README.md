## Final Project

### Question 1:
Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

**Goal:** To classify POI out of enron email dataset. Precision and Recall are greater than 0.3 at least. Machine learnig can create different models to classify and predict dataset. The dataset has multiple features, POI can be related to some important features. However, the dataset also had outliers, we can retrain to remove points with largest residual errors (10%).

Recall is pretty impressive, but accuracy and precisions are pathetically low. Apparantly, Salary is not the only feature useful to identify POIs effectively.

### Initial results:
Try to run the empty one first, it will generate the initial results.

```
tester.dump_classifier_and_data(clf, my_dataset, features_list)
tester.main()
```
```
python poi_id.py  
```

#### Different Classifiers:
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
  



