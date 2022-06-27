list1 = 0
list1 = [1, 2, 3]
list2 = [2, 4 , 5]
list3 = [1, 4, 5]

# from ast import increment_lineno
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('static/bankDetails.csv')

# matplotlib inline

sns.countplot(data=df, x = 'Approved')
plt.plot(list1, list2)
plt.plot(list1, list3)
plt.show()