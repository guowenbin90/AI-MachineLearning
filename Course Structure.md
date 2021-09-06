# Introduction to Machine Learning
[Scikit Learn](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
## Navie Bayes
- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Complement Naive Bayes
- Bernoulli Naive Bayes
- Categorical Naive Baye
- clf = GaussianNB()
## Support Vector Machines
- Classification
- Regression
- sklearn.svm.SVC(kernel='rbf', C=1.0, gamma='scale')
## Decision Trees
- Classification
- Regression
- DecisionTreeClassifier: min_samples_splitint or float, default=2
## Choose Algorithm
- k nearest neighbors: classic, simple, easy to understnd
- adaboot & random forest: 'emsemble methods', meta classifiers built from decision trees
### Process
1. do some research
2. find sklearn documenttion
3. deploy it
4. use it to make predictions
5. evaluate it accuracy
## Datasets and Questions
- Patterns
### Types of Data
- numerical - numerical values (numbers, e.g. salary)  
- categorical - limited number of discrete values (category, e.g. star of movie)
- time series - temporal value (date, timestamp)
- text - words
## Regressions
- Continous supervised learning
- Minimize sum of the squared errors (SSE)
1. Ordinary Least Squares
2. Stochastic Gradient Descent  
- r<sup>2</sup> (r squared)(0<r<sup>2</sup><1): how much of my change in the output (y) is explained by the change in my input  

Comparing Classification & Regression
| Property   |      Supervised classification      |  Regression |
|:----------|:-------------|:------|
| Output type |  Discrete (class labels) | Continous (number) |
| What are you tring to find | Decision boundary | "best fit line" |
| Evaluation | Accuracy | "Sum of squared error" or r<sup>2</sup>("r squared") |

## Outliers
- Ignore:
1. Sensor malfuntions
2. Data entries  
- Pay attention:
3. Freak event
- Removal strategy:
1. Train
2. Remove points with largest residual errors (10%)
3. Retrain
## Clustering
- K-means
1. number of clusters (default:8)
2. max_iter
3. n_init: run with different centroid seeds
## Feature Scaling
rescale = (x-x_min)/(x_max-x_min) (0,1)

These algorithms would be affected by feature rescaling.
1. SVM with RBF kernel
2. K-means clustering  

These algorithms would NOT be affected by feature rescaling.
1. Decision trees
2. Linear regression

## Text Learning
## Feature Selection
## PCA
## Validation
