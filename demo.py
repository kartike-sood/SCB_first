import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas_profiling import ProfileReport



from scipy import stats
df = pd.read_csv("static/bankDetails.csv")
df = df.dropna(axis = 0)
x = df['Monthly_Income']
y = df['Loan_Amount']

# slope, intercept = np.polyfit(x, y, 1)
# if slope > 0:
#     print(f"There is a positive regression between {} and {}")

# q_low = df['Monthly_Income'].quantile(0.1)
# q_hi = df['Loan_Amount'].quantile(0.8)

# first = 'Loan_Amount'
# second = 'Monthly_Income'

# df_filtered = df[(df[second] < q_hi) & (df[first] > q_low)]


var = sns.countplot(x="Approved", data=df)
plt.xlabel("Approved")

total = float(len(df))


            # calculating percentage of different categories in the countplot
for p in var.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()
    y = p.get_height()
    var.annotate(percentage, (x, y),ha='center')

plt.show()
# cat = sns.jointplot(x=first, y=second, height = 7, data=df_filtered,  kind='reg', joint_kws={'line_kws': {'color': 'black'}})
# plt.show()
# print(slope, intercept)


# report = ProfileReport(df)
# report.to_file("for_fun.html")
# """Histogram"""


# q_low = df["Loan_Amount"].quantile(0.25)
# q_hi = df["Monthly_Income"].quantile(0.75)
# df_filtered = df[(df["Loan_Amount"] > q_low) & (df['Monthly_Income'] < q_hi)]

# # plt.hist(df_filtered['Loan_Amount'])
# sns.lmplot(x="Loan_Amount", y="Monthly_Income", data=df_filtered, logistic=True, y_jitter=.03)
# plt.show()


# # x = x[(np.abs(stats.zscore(x)) < 3).all(axis=1)]

# # 
# # q_hi  = df["Loan_Amount"].quantile(0.75)



# # # x = x[x.between(x.quantile(.25), x.quantile(.75))]

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