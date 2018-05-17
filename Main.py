from flask import Flask,jsonify, request

from flasgger import Swagger

from sklearn.externals import joblib

import numpy as np

from flask_cors import CORS

import pandas as  pd

app = Flask(__name__)
Swagger(app)
CORS(app)

@app.route('/input/task', methods=['POST'])
def predict():
    """

    Ini Adalah Endpoint Untuk Memprediksi Minuman

    ---

    tags:

        - Rest Controller

    parameters:

      - name: body

        in: body

        required: true

        schema:

          id: Drinks

          required:

            - Calories

            - Fat

            - Carbohydrates

            - Caffeine

          properties:

            Calories:

              type: int

              description: Please input with valid Calories.

              default: 0

            Fat:

              type: int

              description: Please input with valid Fat.

              default: 0

            Carbohydrates:

              type: int

              description: Please input with valid Carbohydrates.

              default: 0

            Caffeine:

              type: int

              description: Please input with valid Caffeine.

              default: 0

    responses:

        200:

            description: Success Input

    """
    new_task = request.get_json()

    calories = new_task['Calories']
    fat = new_task['Fat']
    carbohydrates = new_task['Carbohydrates']
    caffeine = new_task['Caffeine']
    X_New = np.array([[calories,fat,carbohydrates,caffeine]])

    clf = joblib.load('drinks_pycham.pkl')

    resultPredict = clf[0].predict(X_New)

    return jsonify({'message' : str(resultPredict)})
# app.run(debug=True)
    # return jsonify({'message' : format(clf[1].target_names[resultPredict])})

