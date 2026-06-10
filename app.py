from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("logistic_pkl.pkl", "rb"))

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Heart Failure Prediction</title>
    <style>
        body{
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg,#1e3c72,#2a5298);
            margin:0;
            padding:0;
        }

        .container{
            width:700px;
            margin:40px auto;
            background:white;
            padding:30px;
            border-radius:15px;
            box-shadow:0 0 20px rgba(0,0,0,0.3);
        }

        h1{
            text-align:center;
            color:#1e3c72;
        }

        .grid{
            display:grid;
            grid-template-columns:1fr 1fr;
            gap:15px;
        }

        input{
            width:100%;
            padding:10px;
            border:1px solid #ccc;
            border-radius:8px;
        }

        button{
            width:100%;
            margin-top:20px;
            padding:12px;
            border:none;
            border-radius:8px;
            background:#1e3c72;
            color:white;
            font-size:16px;
            cursor:pointer;
        }

        button:hover{
            background:#2a5298;
        }

        .result{
            margin-top:20px;
            text-align:center;
            font-size:22px;
            font-weight:bold;
        }

        .safe{
            color:green;
        }

        .danger{
            color:red;
        }
    </style>
</head>
<body>

<div class="container">
    <h1>Heart Failure Prediction</h1>

    <form method="POST">

        <div class="grid">

            <input type="number" step="any" name="age" placeholder="Age" required>

            <input type="number" name="anaemia" placeholder="Anaemia (0/1)" required>

            <input type="number" step="any" name="creatinine_phosphokinase" placeholder="Creatinine Phosphokinase" required>

            <input type="number" name="diabetes" placeholder="Diabetes (0/1)" required>

            <input type="number" step="any" name="ejection_fraction" placeholder="Ejection Fraction" required>

            <input type="number" name="high_blood_pressure" placeholder="High Blood Pressure (0/1)" required>

            <input type="number" step="any" name="platelets" placeholder="Platelets" required>

            <input type="number" step="any" name="serum_creatinine" placeholder="Serum Creatinine" required>

            <input type="number" step="any" name="serum_sodium" placeholder="Serum Sodium" required>

            <input type="number" name="sex" placeholder="Sex (0=Female,1=Male)" required>

            <input type="number" name="smoking" placeholder="Smoking (0/1)" required>

            <input type="number" step="any" name="time" placeholder="Follow-up Time" required>

        </div>

        <button type="submit">Predict</button>

    </form>

    {% if prediction %}
    <div class="result {{color}}">
        {{prediction}}
    </div>
    {% endif %}

</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    color = ""

    if request.method == "POST":

        features = [
            float(request.form["age"]),
            float(request.form["anaemia"]),
            float(request.form["creatinine_phosphokinase"]),
            float(request.form["diabetes"]),
            float(request.form["ejection_fraction"]),
            float(request.form["high_blood_pressure"]),
            float(request.form["platelets"]),
            float(request.form["serum_creatinine"]),
            float(request.form["serum_sodium"]),
            float(request.form["sex"]),
            float(request.form["smoking"]),
            float(request.form["time"])
        ]

        prediction_result = model.predict([features])[0]

        if prediction_result == 1:
            prediction = "⚠ High Risk of Death Event"
            color = "danger"
        else:
            prediction = "✅ Low Risk of Death Event"
            color = "safe"

    return render_template_string(
        HTML,
        prediction=prediction,
        color=color
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
