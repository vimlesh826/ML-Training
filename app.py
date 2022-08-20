from flask import Flask, request, jsonify, json
import pickle

app = Flask('ML')


@app.route('/models/predict-salary', methods=["POST"])
def predict_salary():
    input_json = request.get_json(force=True)
    predicted_salary = load_model_and_predict_salary(input_json['experience'])
    if predicted_salary is None:
        predicted_salary = 0
    return common_response(True, 200, predicted_salary)

@app.route('/')
def connected():
    return "Working great ;-)"


def load_model_and_predict_salary(experience: int):
    pickle_file = open('trained-model/trained_model.pkl', 'rb')
    lr_from_pickle = pickle.load(pickle_file)
    prediction_output = lr_from_pickle.predict([[experience]])[0]
    pickle_file.close()
    return prediction_output


def common_response(success: bool = True, response_code: int = 200, data=None):
    response = jsonify(success=success, status=response_code, data=data)
    return response


if __name__ == '__main__':
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=8080)
    app.run(host="0.0.0.0", port=8080)
