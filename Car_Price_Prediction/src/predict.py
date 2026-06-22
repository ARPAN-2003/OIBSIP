import joblib
import pandas as pd

def predict_price(input_dict):

    model = joblib.load(
        "models/car_price_model.pkl"
    )

    sample = pd.DataFrame([input_dict])

    prediction = model.predict(sample)

    return prediction[0]