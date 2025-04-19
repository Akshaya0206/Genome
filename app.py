from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and preprocessors
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define input columns
support_cols = ['CG1_SuppPairs', 'CG2_SuppPairs', 'CC1_SuppPairs', 'CC2_SuppPairs', 'CN1_SuppPairs', 'CN2_SuppPairs']
pval_cols = ['CG1_p_value', 'CG2_p_value', 'CC1_p_value', 'CC2_p_value', 'CN1_p_value', 'CN2_p_value']
all_cols = support_cols + pval_cols

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            input_values = [float(request.form[col]) for col in all_cols]
            input_array = np.array(input_values).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            pred_encoded = model.predict(input_scaled)
            prediction = label_encoder.inverse_transform(pred_encoded)[0]
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template('index.html', support_cols=support_cols, pval_cols=pval_cols, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)