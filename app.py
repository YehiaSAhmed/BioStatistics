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
    x=(207,187,46,4.5,44,0,67,201,31.5,150,74,46,49,0.94)
    pred=d.predection(x)
    if (pred == 0):
        return 'The patient is not diabetic'
    else:
        return render_template('t_diabetic.html')

    

@app.route('/')

def web():
    return render_template('index.html')

if __name__ =="__main__":
    app.run(debug=True)


