# from lib2to3.pgen2.pgen import DFAState
# from pickle import MEMOIZE
from operator import index
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import pandas as pd
import seaborn as sns
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import numpy as np
import datetime

FOLDER = os.path.join('static', 'plotimages')


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = FOLDER

df = 0
list_of_columns = 0
list_of_plots = []
type_of_plot = "xyz"
list_of_categorical_columns = []


ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
UPLOAD_FOLDER = 'static'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

dataframe1 = []

@app.route("/show_columns", methods=['GET', 'POST'])
def show_list_of_columns():
    global list_of_plots
    if request.method == "POST":
        # check if the post request has the file part
        if 'file1' not in request.files:
            return render_template('missing_file.html', warning="I believe you have missed something")

        file = request.files['file1']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return render_template('missing_file.html', warning="I believe you have missed something")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(FOLDER, filename)
            print(filename, filepath)
            file.save(filepath)

            global df
            df = pd.read_csv(filepath)
            global list_of_columns
            list_of_columns = df.columns.to_list()

            list_of_plots = ['Count Plot', 'Scatter Plot',
                             'Histogram', 'Violin Plot', 'Cat Plot']

            global dataframe1
            dataframe1 = df.head(100)
            return render_template("second_page.html", hidden = "visible", dataframe1 = dataframe1, list_of_plots=list_of_plots, list_of_columns=list_of_columns)

@app.route("/fullDataReport", methods = ['GET', 'POST'])
def generateFullReport():
    if request.method == "GET":
        full_report = ProfileReport(df, title = "EDA Report", dark_mode = True, html = {'style' : {'full_width' : True}})
        full_report.to_file("templates/fullReport.html")

        return render_template("fullReport.html")



@app.route("/dataReport", methods = ['GET', 'POST'])
def report():
    global df
    # global list_of_columns
    values = []
    if request.method == "POST":

        values = request.form.to_dict(flat=False)
        print(type(values))
        selected_columns = [value for key, value in values.items()][0]
        # print(selected_columns, "\n")
        df2 = df[selected_columns]

        # correlation1 = df2[selected_columns].corr()
        # correlation1.style.background_gradient(cmap='coolwarm').set_precision(2)

        # plt.matshow(correlation1)
        # plt.show()

        report = ProfileReport(df2, title = "EDA Report", dark_mode = True, html = {'style' : {'full_width' : True}})
        report.to_file("templates/ours.html")

        return render_template("ours.html")
i = 1


@app.route("/countplot", methods=['GET', 'POST'])
def see_cnt_plot():
    if request.method == 'POST':
        global i
        # column = request.form
        values = request.form.to_dict(flat=False)
        print(type(values))
        # print(selected_columns)

        selected_columns = [value for key, value in values.items()]
        print(selected_columns)
        # print(selected_columns, "\n")
        # global df
        df2 = df[selected_columns[0]]
        df2 = pd.melt(df2)
        # print(df2.head())

        var = sns.countplot(x='variable', data=df2, hue='value')
        var = var.get_figure()

        i += 1
        var.savefig(f'static/countplot{i}.png')
        name = f'static/countplot{i}.png'

        return render_template("third_page.html", list_of_plots=list_of_plots, list_of_columns=list_of_columns, name=f'static/countplot{i}.png')


# @app.route('/figure_out', methods = ['GET', 'POST'])
# def graphs():
#     if request.method == 'POST':
#         global i
#         values = request.form.to_dict(flat=False)
#         print(type(values))
#         selected_columns = [value for key, value in values.items()][0]
#         # print(selected_columns, "\n")
#         df2 = df[selected_columns]


#         #   = ProfileReport(df2, title = "EDA Report", dark_mode = True, html = {'style' : {'full_width' : True}})
#         # report.to_file("templates/ours.html")
        # var = sns.countplot(x = df2.columns.to_list()[0], hue = df2.columns.to_list()[1], data = df2)
#         var = var.get_figure()

#         i += 1
#         var.savefig(f'static/scatterplot{i}.png')
#         name = f'static/scatterplot{i}.png'
#         # # plt.savefig("static/mathplt.png")


#         print("Kartike Sood")
#         return render_template("fourth_page.html", list_of_columns = list_of_columns, name = name)

sc = 0


@app.route("/selectcolumnsforplot", methods=['GET', 'POST'])
def columnsForPlot():
    global i
    if i > 1:
       os.remove(os.path.join(FOLDER, f'count{i}.jpg'))

    global sc
    if request.method == 'POST':
        # os.remove(os.path.join(FOLDER, 'count.jpg'))
        inference = ""
        sns.set_theme()
        values = request.form.to_dict(flat=False)
        # print(type(values))

        sc = [value for key, value in values.items()][0]
        # print("The type of sc varible is", type(sc))
        # print(sc)

        fig = plt.gcf()
        fig.set_size_inches(6, 5)

        """I have declared type of plot as a global varibale so that I don not forget which type of plot I am making even when I switch over to another web page"""

        unique_entries = ""
        missing_entries = ""
        total_entries = ""
        my_tab = ""
        if type_of_plot == "Count Plot":

            unique_entries = f"Total number of unique entries in {sc[0]} are {df[sc[0]].nunique()}"
            missing_entries = f"Total number of missing entries in {sc[0]} are {df[sc[0]].isna().sum()}"
            total_entries = f"Total number of entries in {sc[0]} are {len(df[sc[0]]) - df[sc[0]].isna().sum()}"
            my_tab = pd.crosstab(index=df[sc[0]], columns="Frequency", colnames = [" "])


            




            var = sns.countplot(x=sc[0], data=df)
            plt.xlabel(sc[0])


            total = float(len(df))


            # calculating percentage of different categories in the countplot
            for p in var.patches:
                percentage = '{:.2f}%'.format(100 * p.get_height()/total)
                x = p.get_x() + p.get_width()
                y = p.get_height()
                var.annotate(percentage, (x, y),ha='center')





            # plt.ylabel(sc[1])

        elif type_of_plot == "Violin Plot":
            cat = sns.catplot(x=sc[1], y=sc[0], hue="Gender", height=7,
                              aspect=1.5, data=df, kind="violin", split=True)
            plt.xlabel(sc[1])
            plt.ylabel(sc[0])

        elif type_of_plot == "Scatter Plot":
            # print("Kartike Sood")

            # q_low = df["Monthly_Income"].quantile(0.1)
            # q_hi  = df["Loan_Amount"].quantile(0.75)
            q_low = df[sc[1]].quantile(0.1)
            q_hi = df[sc[0]].quantile(0.8)

            first = sc[1]
            second = sc[0]

            df_filtered = df[(df[second] < q_hi) & (df[first] > q_low)]

            # x = x[x.between(x.quantile(.25), x.quantile(.75))]

            # cat = sns.catplot(x="Approved", y="Loan_Amount", hue="Gender", height=7, aspect=1, data=df, kind="violin", split=True)
        #     sns.lmplot(x=first, y=second, data=df_filtered,
        #    logistic=True, y_jitter=.03);
            df_demo = df_filtered.dropna(axis = 0)
            slope, intercept = np.polyfit(df_demo[first], df_demo[second], 1)

            
            if slope > 0:
                inference = f"THERE IS POSITIVE REGRESSION BETWEEN {second} AND {first}"
            elif slope < 0:
                inference = f"THERE IS NEGATIVE REGRESSION BETWEEN {second} AND {first}"
            elif slope == 0:
                inference = f"THERE IS NO REGRESSION BETWEEN {second} AND {first}"
            
            cat = sns.jointplot(x=first, y=second, height = 7, data=df_filtered,  kind='reg', joint_kws={'line_kws': {'color': 'black'}})
            plt.xlabel(first)
            plt.ylabel(second)

        elif type_of_plot == "Histogram":
            q_hi = df[sc[0]].quantile(0.8)
            df_filtered = df[(df[sc[0]] < q_hi)]

            plt.hist(df_filtered[sc[0]])
            plt.xlabel(sc[0])
            plt.ylabel("Density")

        elif type_of_plot == "Cat Plot":
            cat = sns.catplot(x=sc[1], y=sc[0], height=7, aspect=1.2, data=df)
            # plt.show()

            # df_filtered = df[(df[sc[1]] < q_hi) & (df[sc[0]] > q_low)]

            # # x = x[x.between(x.quantile(.25), x.quantile(.75))]

            # # cat = sns.catplot(x="Approved", y="Loan_Amount", hue="Gender", height=7, aspect=1, data=df, kind="violin", split=True)

            # cat = sns.scatterplot(x="Monthly_Income", y="Loan_Amount", data=df_filtered)
        
        i = (i + 1) % 10
        name = os.path.join(FOLDER, f"count{i}.jpg")
        # plt.show()
        plt.savefig(name)

        plt.clf()
        # plt.savefig('static/count.jpg')

        return render_template("fourth_page.html",selected_plot = type_of_plot,my_tab = my_tab, total_entries = total_entries, missing_entries = missing_entries, unique_entries = unique_entries , inference1 = inference, dataframe1 = dataframe1, name=name, columns=list_of_columns, list_of_columns=list_of_columns, list_of_plots=list_of_plots)



@app.route("/edit_columns", methods = {'GET', 'POST'})
def edit():
    return render_template("edit_columns.html", hidden = "hidden", dataframe1 = dataframe1, list_of_plots=list_of_plots, list_of_columns=list_of_columns, columns=list_of_categorical_columns)



@app.route("/changeDF", methods = {'GET', 'POST'})
def changeDF():
    global df
    if request.method == "POST":


        values = request.form.to_dict(flat=False)
        print(values)
        # print(selected_columns)

        sc = [value for key, value in values.items()]
        print(sc)
        if sc[3][0] == '+(Add/concat)':
            newCol = df[sc[1][0]] + df[sc[2][0]]
            df.insert(1, sc[0][0], newCol)
        # df_changed = df[]
        else:
            newCol = df[sc[1][0]] - df[sc[2][0]]
            df.insert(1, sc[0][0], newCol)

        df.to_csv("static/New_Data.csv", index=False)
        dataframe1 = df.head(100)

    return render_template("second_page.html", hidden = "visible", dataframe1 = dataframe1, list_of_plots=list_of_plots, list_of_columns=list_of_columns)



# @app.route("/newfunc", methods=['GET', 'POST'])
# def deriveAge():

#     if request.method == 'POST':
#         values = request.form.to_dict(flat=False)
#         sc = [value for key, value in values.items()][0]
#         col_name = sc[0]


#         global summ
#         global count
#         age = []

#         year = datetime.datetime.today().year

#         for ele in df[sc[0]]:
            
#             if type(ele) == str:
#                 temp = int(ele.split('/')[-1])

#                 temp = temp % 100

#                 if temp <= 30:
#                     temp += 2000
#                 else:
#                     temp += 1900

#                 temp  = year - temp
#                 age.append(temp)

#                 count += 1
#                 summ += temp
#             else:
#                 age.append(-1)

#         df.insert(0, col_name, age)
#         return render_template()


@app.route("/drawplots", methods=['GET', 'POST'])
def usergraphs():

    if request.method == 'POST':

        # column = request.form
        global type_of_plot
        values = request.form.to_dict(flat=False)
        print(type(values))

        """Selected columns is a 2D list"""
        selected_plot = [value for key, value in values.items()][0][0]
        # print(selected_columns)

        global df

        var = 0
        type_of_plot = selected_plot


        # sns.plot(df[])
        # name = os.path.join(FOLDER, "count.jpg")
        # plt.savefig(name)

        if selected_plot == 'Count Plot':

            global list_of_categorical_columns
            list_of_categorical_columns = df.select_dtypes(
                include=['object']).columns.tolist()

            return render_template("count.html", dataframe1 = dataframe1, list_of_plots=list_of_plots, list_of_columns=list_of_columns, columns=list_of_categorical_columns)

        elif selected_plot == 'Scatter Plot':

            list_of_non_categorical_columns = df.select_dtypes(
                exclude=['object']).columns.tolist()

            return render_template("everyotherplot.html", dataframe1 = dataframe1, list_of_plots=list_of_plots, list_of_columns=list_of_columns, columns=list_of_non_categorical_columns)

        elif selected_plot == 'Histogram':
            list_of_numerical_columns = df.select_dtypes(
                include=np.number).columns.to_list()
            return render_template("count.html", list_of_plots=list_of_plots, dataframe1 = dataframe1, list_of_columns=list_of_columns, columns=list_of_numerical_columns)

        # elif selected_plot == "Bar Plot":
        #     list_of_categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

            # return render_template("count.html", list_of_plots = list_of_plots, list_of_columns = list_of_columns, columns = list_of_categorical_columns)
        else:
            return render_template("everyotherplot.html", dataframe1 = dataframe1, list_of_plots=list_of_plots, list_of_columns=list_of_columns, columns=list_of_columns)

    #     if selected_plot == 'Violin Plot':
    #         cat = sns.catplot(x=sc[0], y=sc[1], hue="Gender",
    #         height=7, aspect=1.5, data=df, kind="violin", split=True)

    #         # var = cat.get_figure()

    #     elif selected_plot == 'Histogram':
    #         var = sns.catplot(x="Approved", y="Loan_Amount", hue="Gender",
    #         height=7, aspect=1.5, data=df, kind="violin", split=True)

    #         # var = var.get_figure()

    #     elif selected_plot == 'Bar Plot':
    #         var = sns.catplot(x="Approved", y="Loan_Amount", hue="Gender",
    #         height=7, aspect=1.5, data=df, kind="violin", split=True)

    #         # var = var.get_figure()

    #         # var = var.get_figure()

    #     elif selected_plot == 'Cat Plot':
    #         var = sns.catplot(x="Approved", y="Loan_Amount", hue="Gender",
    #         height=7, aspect=1.5, data=df, kind="violin", split=True)

    #         # var = var.get_figure()

    #     # var.savefig(f'static/violinplot{i}.jpg')

    #     plt.savefig(f'static/plot{i}.jpg')
    #     name = f'static/plot{i}.jpg';

    # return render_template("third_page.html",list_of_columns = list_of_columns, plot = name)


@app.route("/")
def home():
    
    # name = os.path.join(FOLDER, "count.jpg")
    # plt.savefig(name)
    return render_template("home.html")


if __name__ == "__main__":
    app.run()
