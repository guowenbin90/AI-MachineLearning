## Part 2
### Q2:
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

Note that out of all features, except email_address and poi others are numerical. For these numerical values, 'NaN' can be transferred to 0. Remove outliers of the data to pick the features.

```
Decision Tree:
              precision    recall  f1-score   support

     Not PoI       0.79      0.71      0.75        21
         PoI       0.25      0.33      0.29         6

    accuracy                           0.63        27
   macro avg       0.52      0.52      0.52        27
weighted avg       0.67      0.63      0.65        27
```
```
DecisionTreeClassifier()
	Accuracy: 0.71656	Precision: 0.34158	Recall: 0.29700	F1: 0.31773	F2: 0.30496
	Total predictions: 9000	True positives:  594	False positives: 1145	False negatives: 1406	True negatives: 5855
```
