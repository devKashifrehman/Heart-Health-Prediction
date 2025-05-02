from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('heart_disease_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('heart_health.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Data from form 
    age = int(request.form['age'])
    sex = request.form['gender']
    chest_pain = request.form['chestpain']
    resting_bp = int(request.form['restingbp'])
    cholesterol = int(request.form['cholesterol'])
    fasting_bs = int(request.form['fastingbs'])
    resting_ecg = request.form['restingecg']
    max_hr = int(request.form['maxhr'])
    exercise_angina = request.form['exerciseangina']
    oldpeak = float(request.form['oldpeak'])
    st_slope = request.form['stslope']

    # Manual Encoding
    sex = 0 if sex == 'M' else 1
    chest_pain_map = {'ATA':0, 'NAP':1, 'ASY':2, 'TA':3}
    chest_pain = chest_pain_map.get(chest_pain, 0)
    resting_ecg_map = {'Normal':0, 'ST':1, 'LVH':2}
    resting_ecg = resting_ecg_map.get(resting_ecg, 0)
    exercise_angina = 0 if exercise_angina == 'N' else 1
    st_slope_map = {'Up':0, 'Flat':1, 'Down':2}
    st_slope = st_slope_map.get(st_slope, 0)

    # Final input
    final_input = np.array([[age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
                             resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]])

    prediction = model.predict(final_input)[0]

    if prediction == 0:
        message = "🎉 Congratulations! Your Heart is Healthy. Eat a balanced diet: fruits, veggies, fiber-rich foods."
    else:
        if oldpeak > 2.0 or resting_bp > 180:
            message = "⚠️ Critical Risk detected! Please book an appointment with a Cardiologist immediately!"
        else:
            message = "⚠️ Some Heart Risk found. Start a heart-friendly diet: low salt, low sugar, high fiber."

    return render_template('result.html', prediction_text=message)

if __name__ == "__main__":
    app.run(debug=True)
