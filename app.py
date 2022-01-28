from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model1.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['c']
    data2 = request.form['d']
    data3 = request.form['e']
    data4 = request.form['f']
    data5 = request.form['h']
    data6=request.form['j']    

    arr = np.array([[data1, data2, data3, data4,data5,data6]])
    pred = model.predict(arr)
    return render_template('after.html',data=pred)


if __name__ == "__main__":
    app.run(debug=True)
