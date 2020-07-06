from app import app
import pandas as pd
#!flask/bin/python
from flask import Flask, jsonify
from flask import request

import regression_model
from regression_model import pipeline
from regression_model import predict
#0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['GET'])
@app.route('/basket0/<jsonText>')
def get_basket0():
    jsonData = request.args.get('jsonText')
    return "test"

@app.route('/basket/<jsonText>', methods=['GET'])
def get_basket():
    if request.method == 'GET':
        jsonData = request.args.get('jsonText')
        #request.json.get(""
        df = predict.convert_input(jsonData)
        result = predict.make_predict(df)
        predictions = result.get('predictions').tolist()
        return jsonify({'predictions': predictions})

@app.route('/todo/api/v1.0/baskets', methods=['POST'])
def get_baskets():
    if request.method == 'POST':
        jsonData = request.get_json()
        print(jsonData)
        #request.json.get(""
        df = predict.convert_input(jsonData)
        #result = predict.make_predict(df)
        result = predict.make_predict2(df)

        predictions = result #.get('predictions').tolist()
        #return jsonify({'predictions': predictions})
        return predictions

tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web',
        'done': False
    }
]

@app.route('/todo/api/v1.0/tasks', methods=['GET'])
def get_tasks():
    return jsonify({'tasks': tasks})

from flask import abort
@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    task = list(filter(lambda t: t['id'] == task_id, tasks))
    if len(task) == 0:
        abort(404)
    return jsonify({'task': task[0]})

@app.route('/')
@app.route('/index')
def index():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(debug=False, port=8081, host='0.0.0.0') 
