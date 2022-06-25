import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df_html = pd.read_csv('static/train.csv').head().to_html()
text_file = open("table.html", "w")
text_file.write(df_html)
text_file.close()

# sns.countplot(data = df, x = df['state'])