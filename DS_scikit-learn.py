# supervised learning
# unsupervised learning
# reinforced learning (e.g. alpha go)

# <editor-fold desc="Classification"> # alt + cmd + T
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
iris = datasets.load_iris()
type(iris)
print(iris.keys())
type(iris.data)
type(iris.target)
iris.data.shape
iris.data.head()
iris.target_names

X = iris.data
y = iris.target
df = pd.DataFrame(X, columns = iris.feature_names)
df.info()
print(df.head())

# visual EDA
_ = pd.scatter_matrix(df, c = y, figsize = [8, 8], s = 150, marker = 'D') # scatter matrix

plt.figure()
sns.countplot(x='sepal length (cm)', hue=y, data=df, palette='RdBu')
plt.show()

# kNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'], iris['target'])
print(iris['data'].shape)
print(iris['target'].shape)

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))

# [party vote data] Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values
knn.fit(X, y)

# measuring model performance #
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)
knn = KNeighborsClassifier(n_neighbors=8) # smaller k has a risk of over-fitting
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
knn.score(X_test, y_test) # compute accuracy

# example using digits data #
# Import necessary modules
from sklearn import datasets
import matplotlib.pyplot as plt

# Load the digits dataset: digits
digits = datasets.load_digits()
# Print the keys and DESCR of the dataset
print(digits.keys())
print(digits['DESCR'])
# Print the shape of the images and data keys
print(digits.images.shape)
print(digits.data.shape)

# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# Create feature and target arrays
X = digits.data
y = digits.target
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)
# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)
# Fit the classifier to the training data
knn.fit(X_train, y_train)
# Print the accuracy
print(knn.score(X_test, y_test))

# Tuning #
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    # Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
# </editor-fold>

# <editor-fold desc="Regression">
import os
path = "/Users/Ray_Mac/Documents/R Yang/Py_Projects/Datacamp/Datacamp_Data/DS_scikit-learn"
os.chdir(path)
import pandas as pd
boston = pd.read_csv('boston.csv')
import seaborn as sns
sns.heatmap(boston.corr(), square=True, cmap='RdYlGn')
boston.describe
boston.info()
X = boston.drop('MEDV', axis=1).values
y = boston['MEDV'].values
# simple regression
X_rooms = X[:,5]
type(X_rooms), type(y)
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))
X_rooms = X_rooms.reshape(-1, 1)
y = y.reshape(-1, 1)
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))
import numpy as np
from sklearn import linear_model
# from sklearn.linear_model import LinearRegression
reg = linear_model.LinearRegression()
reg.fit(X_rooms, y)
prediction_space = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1, 1)
import matplotlib.pyplot as plt
plt.scatter(X_rooms, y, color='blue')
plt.plot(prediction_space, reg.predict(prediction_space), color='black', linewidth=3)
plt.show()
# multiple regression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
reg_all = linear_model.LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)
reg_all.score(X_test, y_test) # R_square
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))
# k-fold cross-validation
from sklearn.model_selection import cross_val_score
reg = linear_model.LinearRegression()
cv_results = cross_val_score(reg, X, y, cv = 5)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_results)))
np.mean(cv_results)

# time it
%timeit cross_val_score(reg, X, y, cv = 3)
%timeit cross_val_score(reg, X, y, cv = 10)

# regularization to avoid overfitting
from sklearn.linear_model import Ridge # Lasso can shrink non-useful regressor to zero
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
ridge = Ridge(alpha=0.1, normalize=True)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge.score(X_test, y_test)

from sklearn.linear_model import Lasso # this is like a model selection
names = boston.drop('MEDV', axis = 1).columns
lasso = Lasso(alpha = 0.1, normalize=True)
lasso_coef = lasso.fit(X, y).coef_
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation=60)
_ =plt.ylabel('Coefficients')
plt.show()

# Tune Alpha for Ridge
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []
# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)
# Compute scores over range of alphas
for alpha in alpha_space:
    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))
# Display the plot
display_plot(ridge_scores, ridge_scores_std)
# </editor-fold>

# <editor-fold desc="Tuning">
# how good is your model? hyperparameter tuning
# for classification, accuracy may not be a good metric; confusion matrix ROCc checks both Type 1 and Type 2 errors
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=8)
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ROC curve for tuning the thresholds for logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42, stratify=y)
logreg.fit(X_train, y_train)
# lot ROC
from sklearn.metrics import roc_curve
y_pred_prob = logreg.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label = "Logistic Regression")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()
# AUC
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_prob)
# AUC using cross-validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(logreg, X, y, cv = 5, scoring = 'roc_auc')
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_scores))

# Hyperparameter tuning # use cross-validation # tune alpha for ridge/lasso
# tune k for kNN
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X, y)
knn_cv.best_params_
knn_cv.best_score_

# tune C for logit model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space} # param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
logreg_cv.fit(X, y)
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))

# tune bunch of parameters for decision tree model
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}
tree = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)
tree_cv.fit(X,y)
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))

# keep a hold-out set (tune with cv and test with hold-out) for classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
logreg_cv.fit(X_train, y_train)
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))

# keep a hold-out set (tune with cv and test with hold-out) for regression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space} # l1 for Lasso, l2 for Ridge
elastic_net = ElasticNet()
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)
gm_cv.fit(X_train, y_train)
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))
# </editor-fold>

#preprocessing data
##dummy variables
import pandas as pd
df = pd.read_csv('auto.csv')
df_origin = pd.get_dummies(df)
print(df_origin.head()) #3 categories creates 3 dummy, so drop one col#
df_origin = df_origin.drop('origin_Asia', axis=1)
#or
pd.get_dummies(df, drop_first=True)

##missing data
df.insulin.replace(0, np.nan, inplace=True)
df = df.dropna()
###imputing missing value
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
X = imp.transform(X)

from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean',axis=0)
logreg = LogisticRegression()
steps = [('imputation', imp), ('logistic_regression', logreg)]
pipepline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)
pipepline(X_train,y_train)
y_pred = pipepline.predict(X_test)
pipepline.score(X_test, y_test)

