from flask import Flask, render_template, request
'''
import pickle
from sklearn.preprocessing import StandardScaler
'''
import Diabetes_MLwithcleantree as d

# def load_moudel(x):
#     model=pickle.load(open('Predict diabetes using Decision Tree.py','rb'))
#     out= model.predict(np.array(x).reshape(1,-1))
#     return out[0]

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/InputForm.html', methods = ['POST', 'GET'])
def form():
    if request.method == "POST":
        chol_hdl_ratio = (float(request.form['0'])) / (float(request.form['2']))
        waist_hip_ratio = (float(request.form['11'])) / (float(request.form['12']))
        if request.form['5'] == 'Female':
            Gender = 0
        else:
            Gender = 1
        std_data = (request.form['0'], request.form['1'], request.form['2'], chol_hdl_ratio, request.form['4'], Gender,
                    request.form['6'], request.form['7'], request.form['8'], request.form['9'], request.form['10'],
                    request.form['11'], request.form['12'], waist_hip_ratio)
        return result(std_data)
    else:
        return render_template('InputForm.html')
    
    
def result(x):
    pred=d.predection(x)
    if (pred == 0):
        return render_template("f_diabetic.html")
    else:
        return render_template("t_diabetic.html")


if __name__ =="__main__":
    app.run()


