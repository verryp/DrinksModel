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

            - Cholesterol

            - Carbohydrates

            - Sugars

            - Protein

            - Caffeine

          properties:

            Calories:

              type: int

              description: Please input with valid Calories.

              default: 0

            Cholesterol:

              type: int

              description: Please input with valid Cholesterol.

              default: 0

            Carbohydrates:

              type: int

              description: Please input with valid Carbohydrates.

              default: 0

            Sugars:

              type: int

              description: Please input with valid Sugars.

              default: 0

            Protein:

              type: int

              description: Please input with valid Protein.

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
    cholesterol = new_task['Cholesterol']
    carbohydrates = new_task['Carbohydrates']
    sugars = new_task['Sugars']
    protein = new_task['Protein']
    caffeine = new_task['Caffeine']
    X_New = np.array([[calories,cholesterol,carbohydrates,sugars,protein,caffeine]])

    clf = joblib.load('drinks_pycham.pkl')

    resultPredict = clf[0].predict(X_New)

    return jsonify({'message' : str(resultPredict)})
# app.run(debug=True)
    # return jsonify({'message' : format(clf[1].target_names[resultPredict])})

