from pandas import read_csv
# Model Selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# Visualization
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
# Algs
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Metrics
from sklearn.metrics import accuracy_score


# Part one: Load the data
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
names = ['times_pregnant', 'glucose_conc', 'blood_pressure', 'fold_thickness', 'insulin_serum', 'bmi', 'diabetes_predigree_func', 'age', 'class']
data = read_csv(url, names=names)

# Part two: Analyze the data
# print(data.shape)
# print(data.head(20))
# print(data.describe())
# print(data.corr())
# # Class distrubution
# print(data.groupby('class').size())

# Part three: Visualize the data
# a. Univariate plots
# data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
# # b. Multivariate plots
# scatter_matrix(data)
# plt.show()

# Part four: Train, test split
array = data.values
X = array[:, 0:8]
y = array[:, 8]
seed = 7
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# Part five: Testing the models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('SVM', SVC()))

# Part six: Choosing the models
results = []
names = []
seed = 7
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    result = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(result)
    names.append(name)
    message = '%s: %f (%f)' % (name, result.mean(), result.std())
    print(message)

# Part seven: Training the best
# Logistic regression chosen
clf = LogisticRegression()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print(accuracy)
