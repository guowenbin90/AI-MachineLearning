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

Note that out of all features, except email_address and poi others are numerical. For these numerical values, 'NaN' can be transferred to 0.
