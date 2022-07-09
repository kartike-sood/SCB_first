import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy import stats
df = pd.read_csv("static/bankDetails.csv")

"""Histogram"""


q_low = df["Loan_Amount"].quantile(0.75)
df_filtered = df[(df["Loan_Amount"] < q_low)]

plt.hist(df_filtered['Loan_Amount'])
plt.show()


# x = x[(np.abs(stats.zscore(x)) < 3).all(axis=1)]

# 
# q_hi  = df["Loan_Amount"].quantile(0.75)



# # x = x[x.between(x.quantile(.25), x.quantile(.75))]

# # cat = sns.catplot(x="Approved", y="Loan_Amount", hue="Gender", height=7, aspect=1, data=df, kind="violin", split=True)

# cat = sns.scatterplot(x="Monthly_Income", y="Loan_Amount", data=df_filtered)

# plt.show()
# # plt.savefig("kart.jpg")
# c2 = pd.Categorical(["kartike", "sarthak", "aryan", "abhishek", "kartike", "sarthak"])
# print ("\nc2 : ", c2)

# ls3 = [1,2,3,4,5,6,7,8,9,10]
# ls1 = [1, 2, 3, 4]
# ls2 = [i for i in ls1 i is not]

# print(ls1 - ls2)