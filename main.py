from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('ParisHousing.csv')
pipe = pickle.load(open("parisdatestfinal.pkl", 'rb'))

@app.route('/')
def index():
    Size = sorted(data['squareMeters'].unique())
    Rooms = sorted(data['numberOfRooms'].unique())
    Yard = sorted(data['hasYard'].unique())
    Pool = sorted(data['hasPool'].unique())
    Floors = sorted(data['floors'].unique())
    CityCode = sorted(data['cityCode'].unique())
    CityPartRange = sorted(data['cityPartRange'].unique())
    NumPrevOwners = sorted(data['numPrevOwners'].unique())
    Year = sorted(data['made'].unique())
    IsNewBuilt = sorted(data['isNewBuilt'].unique())
    StormProtector = sorted(data['hasStormProtector'].unique())
    Basement = sorted(data['basement'].unique())
    Attic = sorted(data['attic'].unique())
    Garage = sorted(data['garage'].unique())
    StorageRoom = sorted(data['hasStorageRoom'].unique())
    GuestRoom = sorted(data['hasGuestRoom'].unique())
    

    return render_template('index.html', 
                           Size=Size,Rooms=Rooms, Yard=Yard, Pool=Pool, Floors=Floors, 
                           CityCode=CityCode, CityPartRange=CityPartRange, 
                           NumPrevOwners=NumPrevOwners, Year=Year, 
                           IsNewBuilt=IsNewBuilt, StormProtector=StormProtector, 
                           Basement=Basement, Attic=Attic, Garage=Garage, 
                           StorageRoom=StorageRoom, GuestRoom=GuestRoom, 
                           )

@app.route('/predict', methods=['POST'])
def predict():
    input_data = pd.DataFrame([[
        request.form.get('squareMeters'),
        request.form.get('numberOfRooms'),
        request.form.get('hasYard'),
        request.form.get('hasPool'),
        request.form.get('floors'),
        request.form.get('cityCode'),
        request.form.get('cityPartRange'),
        request.form.get('numPrevOwners'),
        request.form.get('made'),
        request.form.get('isNewBuilt'),
        request.form.get('hasStormProtector'),
        request.form.get('basement'),
        request.form.get('attic'),
        request.form.get('garage'),
        request.form.get('hasStorageRoom'),
        request.form.get('hasGuestRoom'),
        
    ]], columns=['squareMeters','numberOfRooms', 'hasYard', 'hasPool', 'floors', 'cityCode', 
                 'cityPartRange', 'numPrevOwners', 'made', 'isNewBuilt', 
                 'hasStormProtector', 'basement', 'attic', 'garage', 
                 'hasStorageRoom', 'hasGuestRoom'])

    # Convert data to appropriate types
    input_data = input_data.astype({
        'squareMeters' :int,'numberOfRooms': int, 'hasYard': int, 'hasPool': int, 'floors': int, 
        'cityCode': int, 'cityPartRange': int, 'numPrevOwners': int, 
        'made': int, 'isNewBuilt': int, 'hasStormProtector': int, 
        'basement': int, 'attic': int, 'garage': int, 
        'hasStorageRoom': int, 'hasGuestRoom': int, 
    })

    prediction = pipe.predict(input_data)[0]

    return str(prediction)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
