from pandas import DataFrame
import numpy

myarray = numpy.array([[1, 2, 3], [4, 5, 6]])
row_names = ['a', 'b']
col_names = ['one', 'two', 'three']
df = DataFrame(myarray, index=row_names, columns=col_names)

print(df)
