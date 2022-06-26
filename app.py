from flask import Flask,request,jsonify
import joblib

app = Flask(__name__)
import sklearn

model = joblib.load('testmodel.pkl')

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/titanicModel', methods=['POST'])
def titanicModel():
    if request.method == 'POST':
        some_json = request.get_json()
        data = some_json['body']
        prediction = model.predict(data)
        pred = prediction.tolist()
        print(type(pred))
        return jsonify({'prediction': pred})



if __name__ == '__main__':
    app.run()
