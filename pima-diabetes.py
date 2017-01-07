from pandas import read_csv
from sklearn.model_selection import train_test_split
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt

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
