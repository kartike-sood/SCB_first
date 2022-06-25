# from lib2to3.pgen2.pgen import DFAState
# from pickle import MEMOIZE
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import pandas as pd
import seaborn as sns
from pandas_profiling import ProfileReport
df = 0
list_of_columns = 0
app = Flask(__name__)

ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
UPLOAD_FOLDER = 'static'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/show_columns", methods=['GET', 'POST'])
def show_list_of_columns():

    if request.method == "POST":
        # check if the post request has the file part
        if 'file1' not in request.files:
            return render_template('missing_file.html', warning = "I believe you have missed something")

        file = request.files['file1']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return render_template('missing_file.html', warning = "I believe you have missed something")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('static', filename)
            print(filename, filepath)
            file.save(filepath)

            global df
            df = pd.read_csv(filepath)
            global list_of_columns
            list_of_columns = df.columns.to_list()
            

            return render_template("second_page.html", list_of_columns = list_of_columns)

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


        report = ProfileReport(df2, title = "EDA Report", dark_mode = True, html = {'style' : {'full_width' : True}})
        report.to_file("templates/ours.html")

        return render_template("ours.html")


@app.route("/countplot", methods = ['GET', 'POST'])
def see_cnt_plot():
    if request.method == 'POST':
        # column = request.form
        values = request.form.to_dict(flat=False)
        print(type(values))
        selected_columns = [value for key, value in values.items()][0]
        # print(selected_columns, "\n")
        # global df
        df2 = df[selected_columns]

        var = sns.countplot(x = selected_columns[0], data = df2)
        var = var.get_figure()

        name = 'static/countplot.png'
        var.savefig(name)


        return render_template("third_page.html", list_of_columns = list_of_columns, name = name)



@app.route("/")
def home():
    return render_template("home.html")

if __name__ == "__main__":
    app.run()