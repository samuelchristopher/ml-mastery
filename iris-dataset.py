# Libraries
import numpy
import scipy
import sklearn
import pandas
import matplotlib
# Visualization
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
# Metrics
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# Algs
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

# Part one: Load the data
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names = ['sepal_lenght (cm)', 'sepal_width (cm)', 'petal_lenght (cm)', 'petal_width (cm)', 'class']
data = pandas.read_csv(url, names=names)

# # Part two: Analyze the data
# # a. Dimensions
# print(data.shape)
# # b. Look at the data
# print(data.head(20))
# # c. Statistical summary
# print(data.describe())
# # d. Class distribution
# print(data.groupby('class').size())

# Part three: Visualize the data
# a. Univariate plot
# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# data.hist()
# plt.show()

# b. Multivariate plot
# scatter_matrix(data)
# plt.show()

# Part four: Train, test split
array = data.values
X = array[:, 0:4]
y = array[:, 4]
test_size = 0.2
seed = 7
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=seed)

# Part five: Test harness
seed = 7
scoring = 'accuracy'

# Part six: Modeling
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('NB', GaussianNB()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC()))

results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_result = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_result)
    names.append(name)
    message = '%s: %f (%f)' % (name, cv_result.mean(), cv_result.std())
    print(message)

# Part seven: Comparing algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Part eight: Predicting using the best model
# SVM model was chosen
svm = SVC()
svm.fit(X_train, y_train)
pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)
cr = classification_report(y_test, pred)
print(accuracy)
print(cm)
print(cr)
