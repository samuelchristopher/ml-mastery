from pandas import read_csv

# Part one: Load the data
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
names = ['times_pregnant', 'glucose_conc', 'blood_pressure', 'fold_thickness', 'insulin_serum', 'bmi', 'diabetes_predigree_func', 'age', 'class']
data = read_csv(url, names=names)

# Part two: Analyze the data
print(data.shape)
print(data.head(20))
print(data.describe())
print(data.corr())
# Class distrubution
print(data.groupby('class').size())
