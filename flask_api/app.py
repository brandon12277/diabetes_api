from flask import Flask,jsonify,request
import joblib
from flask_cors import CORS

app = Flask(__name__)
model = joblib.load('diabetes.joblib')
sc = joblib.load('scaler.joblib')
CORS(app)

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/api',methods=['POST'])
def api():
    try:
        data = request.get_json()

        data_val = [list(data.values())]
        print(data_val)
        
        data_val = sc.transform(data_val)
        print(data_val)

        predict = model.predict(data_val)[0]
        print(predict)

        return jsonify({'class' : str(predict)})

    except Exception as e:
        return jsonify({'error': str(e)})
   

if __name__ == '__main__':
    app.run(debug=True)
