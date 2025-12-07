from flask import Flask,request, render_template, send_file, send_from_directory
import numpy as np
import pickle
from flask_pymongo import PyMongo
from dotenv import load_dotenv
import os
import json

# additional imports for unified crop recommendation
from pathlib import Path


load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")

#loading models
Jmodel = pickle.load(open('jmodel.pkl','rb'))
Wmodel = pickle.load(open('wmodel.pkl','rb'))
Cmodel = pickle.load(open('cmodel.pkl','rb'))
Smodel = pickle.load(open('smodel.pkl','rb'))
Bmodel = pickle.load(open('bmodel.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))

# ================= Unified Crop Recommendation Model Loading =================
# Expecting models from sibling project: Crop_Recommendation-main
BASE_DIR = Path(__file__).resolve().parent
RECO_DIR = (BASE_DIR.parent.parent / 'Crop_Recommendation-main')
try:
    rec_model = pickle.load(open(str(RECO_DIR / 'model.pkl'), 'rb'))
    rec_stand_scaler = pickle.load(open(str(RECO_DIR / 'standscaler.pkl'), 'rb'))
    rec_minmax_scaler = pickle.load(open(str(RECO_DIR / 'minmaxscaler.pkl'), 'rb'))
except Exception as _:
    rec_model = None
    rec_stand_scaler = None
    rec_minmax_scaler = None

#flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# app.config["MONGO_URI"] = os.getenv("MONGO_URI")
app.config["MONGO_URI"] = MONGO_URL
mongo = PyMongo(app)

collection = mongo.db.crop_statistics

@app.route('/')
def index():
    # Serve the polished landing page from the root index folder
    try:
        root_index_path = (Path(__file__).resolve().parent.parent.parent / 'index' / 'index.html')
        return send_file(str(root_index_path))
    except Exception:
        # Fallback to existing backend index if root landing is unavailable
        return render_template('index.html')

@app.route('/landing-assets/<path:filename>')
def landing_assets(filename):
    # Serve CSS/JS assets that live alongside the landing index
    root_assets_dir = Path(__file__).resolve().parent.parent.parent / 'index'
    return send_from_directory(str(root_assets_dir), filename)

@app.route('/home')
def home():
    return render_template('homepage.html')    

@app.route('/submitCropData')
def submitcropdata():
    return render_template('submitCropDetails.html')  

@app.route('/cropDetailsSubmited',methods=['POST'])
def cropdatasubmited():
    if request.method == 'POST':
        commodity = request.form['commodityname']
        district = request.form['district']
        crop_data = {
        "commodity": commodity,
        "district": district
        }
        collection.insert_one(crop_data)
        return render_template('cropDataSubmit.html')


@app.route('/predict')
def predict():
    return render_template('predict.html')  

# ======================== Crop Recommendation Routes =========================
@app.route('/recommend')
def recommend_form():
    return render_template('recommend.html')


@app.route('/recommend/predict', methods=['POST'])
def recommend_predict():
    if rec_model is None or rec_stand_scaler is None or rec_minmax_scaler is None:
        return ("Recommendation model not available. Ensure 'Crop_Recommendation-main' exists with model files.", 500)

    # Parse and validate numeric inputs
    try:
        N = float(request.form.get('Nitrogen', '').strip())
        P = float(request.form.get('Phosporus', '').strip())
        K = float(request.form.get('Potassium', '').strip())
        temp = float(request.form.get('Temperature', '').strip())
        humidity = float(request.form.get('Humidity', '').strip())
        ph = float(request.form.get('pH', '').strip())
        rainfall = float(request.form.get('Rainfall', '').strip())
    except Exception:
        return ("Invalid inputs. Please enter numeric values for N, P, K, Temperature, Humidity, pH, Rainfall.", 400)

    # Build float feature vector in expected order
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array([feature_list], dtype=np.float64)

    try:
        mx_features = rec_minmax_scaler.transform(single_pred)
        sc_mx_features = rec_stand_scaler.transform(mx_features)
        prediction = rec_model.predict(sc_mx_features)
    except Exception as _:
        return ("Failed to transform or predict. Check input ranges and model compatibility.", 500)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result_text = f"{crop} is the best crop to be cultivated right there"
    else:
        result_text = "Sorry, we could not determine the best crop with the provided data."

    return render_template('recommend.html', result=result_text)

@app.route('/crop_statistics',methods=['POST','GET'])
def crop_statistics():
    
    distlist = ['Solapur','Nanded','Buldhana','Amravati','Sambhajinagar']
    commoditylist = ['Jowar','Bajara','Cotton','Sugarcane','Wheat']

    cursor  = collection.find({})
    data = list(cursor)
  
    crop_per_dist = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
    
    for dataele in data:
        commodityindex = commoditylist.index(dataele.get('commodity'))
        distindex = distlist.index(dataele.get('district'))
        crop_per_dist[distindex][commodityindex] = crop_per_dist[distindex][commodityindex] + 1

    crop_state = [0,0,0,0,0]
    i=0
    for ele in crop_per_dist:
        for ele_value in ele:
            crop_state[i] = crop_state[i] + ele_value
            i = i+1
        i =0
   
    return render_template("crop_statistics.html",crop_state_data = json.dumps(crop_state))
    
@app.route('/Jowar')
def jowar():
    cursor  = collection.find({'commodity':"Jowar"})
    data = list(cursor)
   
    distlist = ['Solapur','Nanded','Buldhana','Amravati','Sambhajinagar']
    frequency_count = [0,0,0,0,0]

    distcount = 0
    for district in distlist:
        for dist in data:
            if(dist.get('district') == district):
                frequency_count[distcount] = frequency_count[distcount] + 1
        distcount = distcount + 1

    return render_template('commodity.html',ID="Jowar",commodity_data=json.dumps(frequency_count))   


@app.route('/Bajara')
def bajara():
    cursor  = collection.find({'commodity':"Bajara"})
    data = list(cursor)

    distlist = ['Solapur','Nanded','Buldhana','Amravati','Sambhajinagar']
    frequency_count = [0,0,0,0,0]

    distcount = 0
    for district in distlist:
        for dist in data:
            if(dist.get('district') == district):
                frequency_count[distcount] = frequency_count[distcount] + 1
        distcount = distcount + 1

    return render_template('commodity.html',ID="Bajara",commodity_data=json.dumps(frequency_count))   


@app.route('/Cotton')
def cotton():

    cursor  = collection.find({'commodity':"Cotton"})
    data = list(cursor)

    distlist = ['Solapur','Nanded','Buldhana','Amravati','Sambhajinagar']
    frequency_count = [0,0,0,0,0]

    distcount = 0
    for district in distlist:
        for dist in data:
            if(dist.get('district') == district):
                frequency_count[distcount] = frequency_count[distcount] + 1
        distcount = distcount + 1

    return render_template('commodity.html',ID="Cotton",commodity_data=json.dumps(frequency_count))   


@app.route('/Sugarcane')
def sugarcane():

    cursor  = collection.find({'commodity':"Sugarcane"})
    data = list(cursor)

    distlist = ['Solapur','Nanded','Buldhana','Amravati','Sambhajinagar']
    frequency_count = [0,0,0,0,0]

    distcount = 0
    for district in distlist:
        for dist in data:
            if(dist.get('district') == district):
                frequency_count[distcount] = frequency_count[distcount] + 1
        distcount = distcount + 1

    return render_template('commodity.html',ID="Sugarcane",commodity_data=json.dumps(frequency_count))   


@app.route('/Wheat')
def wheat():

    cursor  = collection.find({'commodity':"Wheat"})
    data = list(cursor)

    distlist = ['Solapur','Nanded','Buldhana','Amravati','Sambhajinagar']
    frequency_count = [0,0,0,0,0]

    distcount = 0
    for district in distlist:
        for dist in data:
            if(dist.get('district') == district):
                frequency_count[distcount] = frequency_count[distcount] + 1
        distcount = distcount + 1

    return render_template('commodity.html',ID="Wheat",commodity_data=json.dumps(frequency_count))   




@app.route('/Solapur')
def solapur():
     
    cursor  = collection.find({'district':"Solapur"})
    data = list(cursor)

    commoditylist = ['Jowar','Bajara','Cotton','Sugarcane','Wheat']
    crop_frequency = [0,0,0,0,0]
    
    
    for dataele in data:
        commodityindex = commoditylist.index(dataele.get('commodity'))
        crop_frequency[commodityindex] = crop_frequency[commodityindex] + 1

    return render_template('district.html',ID="Solapur",crop_frequency=json.dumps(crop_frequency))  

@app.route('/Nanded')
def nanded():
     
    cursor  = collection.find({'district':"Nanded"})
    data = list(cursor)

    commoditylist = ['Jowar','Bajara','Cotton','Sugarcane','Wheat']
    crop_frequency = [0,0,0,0,0]
    
    
    for dataele in data:
        commodityindex = commoditylist.index(dataele.get('commodity'))
        crop_frequency[commodityindex] = crop_frequency[commodityindex] + 1

    return render_template('district.html',ID="Nanded",crop_frequency=json.dumps(crop_frequency))  


@app.route('/Buldhana')
def buldhana():
     
    cursor  = collection.find({'district':"Buldhana"})
    data = list(cursor)

    commoditylist = ['Jowar','Bajara','Cotton','Sugarcane','Wheat']
    crop_frequency = [0,0,0,0,0]
    
    
    for dataele in data:
        commodityindex = commoditylist.index(dataele.get('commodity'))
        crop_frequency[commodityindex] = crop_frequency[commodityindex] + 1

    return render_template('district.html',ID="Buldhana",crop_frequency=json.dumps(crop_frequency))  

@app.route('/Amaravati')
def amaravati():
     
    cursor  = collection.find({'district':"Amravati"})
    data = list(cursor)

    commoditylist = ['Jowar','Bajara','Cotton','Sugarcane','Wheat']
    crop_frequency = [0,0,0,0,0]
    
    
    for dataele in data:
        commodityindex = commoditylist.index(dataele.get('commodity'))
        crop_frequency[commodityindex] = crop_frequency[commodityindex] + 1

    return render_template('district.html',ID="Amaravati",crop_frequency=json.dumps(crop_frequency))  

@app.route('/Sambhajinagar')
def sambhajinagar():
     
    cursor  = collection.find({'district':"Sambhajinagar"})
    data = list(cursor)

    commoditylist = ['Jowar','Bajara','Cotton','Sugarcane','Wheat']
    crop_frequency = [0,0,0,0,0]
    
    
    for dataele in data:
        commodityindex = commoditylist.index(dataele.get('commodity'))
        crop_frequency[commodityindex] = crop_frequency[commodityindex] + 1

    return render_template('district.html',ID="Sambhajinar",crop_frequency=json.dumps(crop_frequency))  


@app.route('/result',methods=['POST'])
def result():
    if request.method == 'POST':
        # Normalize and validate inputs
        commodity_input = str(request.form.get('commodityname', '')).strip().lower()
        commodity_map = {
            'jowar': 'Jowar',
            'wheat': 'Wheat',
            'cotton': 'Cotton',
            'sugarcane': 'Sugarcane',
            'bajra': 'Bajara',   # common spelling
            'bajara': 'Bajara'   # backend model key
        }
        commoditytype = commodity_map.get(commodity_input)

        # Coerce numeric fields safely
        try:
            month = int(request.form.get('month', 0))
            Year = int(request.form.get('year', 0))
            average_rain_fall = float(request.form.get('average_rain_fall', 0))
        except Exception:
            return ("Invalid numeric inputs. Please provide: month (1-12), year (e.g. 2025), rainfall (mm)", 400)

        if commoditytype is None:
            return ("Unsupported commodity. Use one of: Jowar, Wheat, Bajra, Sugarcane, Cotton", 400)

        if month < 1 or month > 12 or Year < 1900:
            return ("Invalid month/year. Month must be 1-12 and year must be >= 1900", 400)

        NextYear = Year + 1

        avgPriceyear = []
        mspyear = []
        mspnextyear = []
        avgPriceNextyear = []
        months_labels = ["jan","Feb","March","April","May","June","July","Aug","Sept","Oct","Nov","Dec"]
        monthcount = 1
        yearcount = 2021

        rainfall2024 = [1,2,3,1,8,673,1318,779,408,106,44,8]
        rainfall2023 = [90,7.2,29.9,41.4,67.6,148.6,315.9,162.7,190.3,50.8,9.3,8]
        x_count =0 

        cropimges = ['/static/images/jowarlogo.webp',
                     '/static/images/wheatlogo.avif',
                     '/static/images/cottonlogo.jpg',
                     '/static/images/sugarcanelogo.jpg',
                     '/static/images/bajralogo.jpg'
                    ]
        

        features = np.array([[month,Year,average_rain_fall]],dtype=object)
        transformed_features = preprocessor.transform(features)

        if(commoditytype == "Jowar"):
            cropface = cropimges[0]
            prediction = Jmodel.predict(transformed_features).reshape(1,-1)
            predicted_value = round(prediction[0][0] , 3)
            min_value = round((predicted_value*1550)/100,2)
            max_value = round((predicted_value*2970)/100,2)
            avg_value = round((min_value + max_value) / 2,2)

            for x in rainfall2023:
                features = np.array([[monthcount,yearcount,x]],dtype=object)
                transformed_features = preprocessor.transform(features)
                prediction = Jmodel.predict(transformed_features).reshape(1,-1)
                predicted_value = round(prediction[0][0] , 3)
                predictionMsp = round((predicted_value*2970)/100,2)
                mspyear.append(predictionMsp)
                predictionAvg = round((predicted_value*1550)/100,2)
                avgPriceyear.append(predictionAvg)
                monthcount = monthcount + 1
                x_count = x_count + 1

            for x in rainfall2024:
                features = np.array([[monthcount,yearcount,x]],dtype=object)
                transformed_features = preprocessor.transform(features)
                prediction = Jmodel.predict(transformed_features).reshape(1,-1)
                predicted_value = round(prediction[0][0] , 3)
                predictionMsp = round((predicted_value*2970)/100,2)
                mspnextyear.append(predictionMsp)
                predictionAvg = round((predicted_value*1550)/100,2)
                avgPriceNextyear.append(predictionAvg)
                x_count = x_count + 1

        elif(commoditytype == "Wheat"):
            cropface = cropimges[1]
            prediction = Wmodel.predict(transformed_features).reshape(1,-1)
            predicted_value = round(prediction[0][0] , 3)
            min_value = round((predicted_value*1350)/100,2)
            max_value = round((predicted_value*2125)/100,2)
            avg_value = round((min_value + max_value) / 2 ,2)

            for x in rainfall2023:
                features = np.array([[monthcount,yearcount,x]],dtype=object)
                transformed_features = preprocessor.transform(features)
                prediction = Wmodel.predict(transformed_features).reshape(1,-1)
                predicted_value = round(prediction[0][0] , 3)
                predictionMsp = round((predicted_value*2125)/100,2)
                mspyear.append(predictionMsp)
                predictionAvg = round((predicted_value*1350)/100,2)
                avgPriceyear.append(predictionAvg)
                monthcount = monthcount + 1
                x_count = x_count + 1

            for x in rainfall2024:
                features = np.array([[monthcount,yearcount,x]],dtype=object)
                transformed_features = preprocessor.transform(features)
                prediction = Wmodel.predict(transformed_features).reshape(1,-1)
                predicted_value = round(prediction[0][0] , 3)
                predictionMsp = round((predicted_value*2970)/100,2)
                mspnextyear.append(predictionMsp)
                predictionAvg = round((predicted_value*2125)/100,2)
                avgPriceNextyear.append(predictionAvg)
                x_count = x_count + 1
        
        elif(commoditytype == "Cotton"):
            cropface = cropimges[2]
            prediction = Cmodel.predict(transformed_features).reshape(1,-1)
            predicted_value = round(prediction[0][0] , 3)
            min_value = round((predicted_value*3600)/100,2)
            max_value = round((predicted_value*6080)/100,2)
            avg_value = round((min_value + max_value) / 2 ,2)

            for x in rainfall2023:
                features = np.array([[monthcount,yearcount,x]],dtype=object)
                transformed_features = preprocessor.transform(features)
                prediction = Cmodel.predict(transformed_features).reshape(1,-1)
                predicted_value = round(prediction[0][0] , 3)
                predictionMsp = round((predicted_value*6080)/100,2)
                mspyear.append(predictionMsp)
                predictionAvg = round((predicted_value*3600)/100,2)
                avgPriceyear.append(predictionAvg)
                monthcount = monthcount + 1
                x_count = x_count + 1

            for x in rainfall2024:
                features = np.array([[monthcount,yearcount,x]],dtype=object)
                transformed_features = preprocessor.transform(features)
                prediction = Cmodel.predict(transformed_features).reshape(1,-1)
                predicted_value = round(prediction[0][0] , 3)
                predictionMsp = round((predicted_value*6080)/100,2)
                mspnextyear.append(predictionMsp)
                predictionAvg = round((predicted_value*3600)/100,2)
                avgPriceNextyear.append(predictionAvg)
                x_count = x_count + 1
        
        elif(commoditytype == "Sugarcane"):
            cropface = cropimges[3]
            prediction = Smodel.predict(transformed_features).reshape(1,-1)
            predicted_value = round(prediction[0][0] , 3)
            min_value = round((predicted_value*2250)/100,2)
            max_value = round((predicted_value*2775)/100,2)
            avg_value = round((min_value + max_value) / 2 ,2)

            for x in rainfall2023:
                features = np.array([[monthcount,yearcount,x]],dtype=object)
                transformed_features = preprocessor.transform(features)
                prediction = Smodel.predict(transformed_features).reshape(1,-1)
                predicted_value = round(prediction[0][0] , 3)
                predictionMsp = round((predicted_value*2775)/100,2)
                mspyear.append(predictionMsp)
                predictionAvg = round((predicted_value*2250)/100,2)
                avgPriceyear.append(predictionAvg)
                monthcount = monthcount + 1
                x_count = x_count + 1

            for x in rainfall2024:
                features = np.array([[monthcount,yearcount,x]],dtype=object)
                transformed_features = preprocessor.transform(features)
                prediction = Smodel.predict(transformed_features).reshape(1,-1)
                predicted_value = round(prediction[0][0] , 3)
                predictionMsp = round((predicted_value*2970)/100,2)
                mspnextyear.append(predictionMsp)
                predictionAvg = round((predicted_value*2775)/100,2)
                avgPriceNextyear.append(predictionAvg)
                x_count = x_count + 1
        
        elif(commoditytype == "Bajara"):
            cropface = cropimges[4]
            prediction = Bmodel.predict(transformed_features).reshape(1,-1)
            predicted_value = round(prediction[0][0] , 3)
            min_value = round((predicted_value*1175)/100,2)
            max_value = round((predicted_value*2350)/100,2)
            avg_value = round((min_value + max_value) / 2 ,2)

            for x in rainfall2023:
                features = np.array([[monthcount,yearcount,x]],dtype=object)
                transformed_features = preprocessor.transform(features)
                prediction = Bmodel.predict(transformed_features).reshape(1,-1)
                predicted_value = round(prediction[0][0] , 3)
                predictionMsp = round((predicted_value*2350)/100,2)
                mspyear.append(predictionMsp)
                predictionAvg = round((predicted_value*1175)/100,2)
                avgPriceyear.append(predictionAvg)
                monthcount = monthcount + 1
                x_count = x_count + 1

            for x in rainfall2024:
                features = np.array([[monthcount,yearcount,x]],dtype=object)
                transformed_features = preprocessor.transform(features)
                prediction = Bmodel.predict(transformed_features).reshape(1,-1)
                predicted_value = round(prediction[0][0] , 3)
                predictionMsp = round((predicted_value*2970)/100,2)
                mspnextyear.append(predictionMsp)
                predictionAvg = round((predicted_value*2350)/100,2)
                avgPriceNextyear.append(predictionAvg)
                x_count = x_count + 1



        # Guard against unexpected empty lists
        if not mspyear or not avgPriceyear:
            return ("Unable to compute prediction series. Please check inputs and try again.", 500)

        maxmspyear = max(mspyear)
        maxavgPriceyear = max(avgPriceyear)

        goldmonthindex = mspyear.index(maxmspyear) + 1

        minmspyear = min(mspyear)
        minavgPriceyear = min(avgPriceyear)

        silvermonthindex = mspyear.index(minmspyear) + 1
        


        return render_template('result.html',prediction = predicted_value,
                                             cropface = cropface,
                                             min_value = min_value,
                                             max_value = max_value,
                                             avg_value= avg_value,
                                             year = Year,
                                             NextYear = NextYear,
                                             month = month,
                                             maxhigh = maxmspyear,
                                             maxlow = maxavgPriceyear,
                                             minhigh = minmspyear,
                                             minlow = minavgPriceyear,
                                             goldmonth = goldmonthindex,
                                             silvermonth = silvermonthindex,
                                             months_labels = months_labels,
                                             mspyear = json.dumps(mspyear),
                                             minPriceYear = json.dumps(avgPriceyear),
                                             mspnextyear = json.dumps(mspnextyear),
                                             minPriceNextYear = json.dumps(avgPriceNextyear)
                                             )

if __name__=="__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)