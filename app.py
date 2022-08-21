from flask import Flask, request, jsonify, render_template
import pickle

application = Flask('ML')


@application.route('/models/predict-salary', methods=["POST"])
def predict_salary():
    input_json = request.get_json(force=True)
    predicted_salary = load_model_and_predict_salary(input_json['experience'])
    if predicted_salary is None:
        predicted_salary = 0
    return common_response(True, 200, predicted_salary, "Salary predicted successfully.")


@application.route('/')
def connected():
    return render_template('index.html')
    # return "Working great ;-)"


def load_model_and_predict_salary(experience: int):
    pickle_file = open('trained-model/trained_model.pkl', 'rb')
    lr_from_pickle = pickle.load(pickle_file)
    prediction_output = lr_from_pickle.predict([[experience]])[0]
    pickle_file.close()
    return round(prediction_output, 2)


def common_response(success: bool = True, response_code: int = 200, data=None, message="Success"):
    response = jsonify(success=success, status=response_code, data=data, message=message)
    return response


if __name__ == '__main__':
    application.run(host="0.0.0.0", port=80)
    # application.run(host="127.0.0.1", port=5000)
