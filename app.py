from flask import Flask, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import Diabetes_MLwithcleantree as d

# def load_moudel(x):
#     model=pickle.load(open('Predict diabetes using Decision Tree.py','rb'))
#     out= model.predict(np.array(x).reshape(1,-1))
#     return out[0]

app = Flask(__name__)

@app.route('/predict')
def predict():
    x=(193,77,49,3.9,19,0,61,119,22.5,118,70,32,38,0.84)
    pred=d.predection(x)
    if (pred == 0):
        return render_template('f_diabetic.html')
    else:
        return render_template('t_diabetic.html')

    

@app.route('/')

def web():
    return render_template('index.html')

if __name__ =="__main__":
    app.run(debug=True)


