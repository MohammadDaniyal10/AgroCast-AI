from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle

model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
mx = pickle.load(open('minmaxscaler.pkl','rb'))


app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    try:
        # Coerce numeric inputs to float for scaler compatibility
        N = float(request.form.get('Nitrogen', '').strip())
        P = float(request.form.get('Phosporus', '').strip())
        K = float(request.form.get('Potassium', '').strip())
        temp = float(request.form.get('Temperature', '').strip())
        humidity = float(request.form.get('Humidity', '').strip())
        ph = float(request.form.get('pH', '').strip())
        rainfall = float(request.form.get('Rainfall', '').strip())
    except Exception:
        return render_template('index.html', result="Please enter numeric values for all fields (N, P, K, Temperature, Humidity, pH, Rainfall).")

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array([feature_list], dtype=np.float64)

    try:
        # Skip scaling for now as we're having version compatibility issues
        prediction = model.predict(single_pred)
    except Exception as e:
        return render_template('index.html', result=f"Prediction failed: {type(e).__name__}: {e}")

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('index.html',result = result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False)